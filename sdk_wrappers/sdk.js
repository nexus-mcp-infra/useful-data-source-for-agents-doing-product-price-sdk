
```javascript
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const BUYWHERE_API_BASE = process.env.BUYWHERE_API_BASE || 'https://api.buywhere-sg.nexus.io/v1';
const DEFAULT_TIMEOUT_MS = 30000;
const DEFAULT_MAX_RETRIES = 3;
const RETRY_BACKOFF_BASE_MS = 500;

class BuyWhereSGError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'BuyWhereSGError';
    this.statusCode = statusCode || null;
    this.body = body || null;
  }
}

class AuthenticationError extends BuyWhereSGError {
  constructor(message) {
    super(message, 401, null);
    this.name = 'AuthenticationError';
  }
}

class RateLimitError extends BuyWhereSGError {
  constructor(retryAfterSeconds) {
    super(`Rate limit exceeded. Retry after ${retryAfterSeconds}s`, 429, null);
    this.name = 'RateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

class ValidationError extends BuyWhereSGError {
  constructor(message) {
    super(message, 400, null);
    this.name = 'ValidationError';
  }
}

const VALID_PRODUCT_DOMAINS = new Set(['electronics', 'appliances', 'computing']);
const VALID_VENDORS = new Set(['lazada', 'shopee', 'courts', 'harvey_norman', 'all']);
const VALID_AVAILABILITY = new Set(['online', 'physical', 'both']);
const VALID_SORT_BY = new Set(['price_asc', 'price_desc', 'dispersion_desc', 'freshness_desc']);

function validateNonEmptyString(value, fieldName) {
  if (value === null || value === undefined) {
    throw new ValidationError(`${fieldName} is required and cannot be null or undefined`);
  }
  if (typeof value !== 'string') {
    throw new ValidationError(`${fieldName} must be a string, got ${typeof value}`);
  }
  if (value.trim().length === 0) {
    throw new ValidationError(`${fieldName} cannot be an empty string`);
  }
}

function validatePositiveInteger(value, fieldName, min, max) {
  if (value === null || value === undefined) return;
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new ValidationError(`${fieldName} must be an integer, got ${typeof value}`);
  }
  if (min !== undefined && value < min) {
    throw new ValidationError(`${fieldName} must be >= ${min}, got ${value}`);
  }
  if (max !== undefined && value > max) {
    throw new ValidationError(`${fieldName} must be <= ${max}, got ${value}`);
  }
}

function validateEnum(value, fieldName, validSet) {
  if (value === null || value === undefined) return;
  if (!validSet.has(value)) {
    throw new ValidationError(
      `${fieldName} must be one of [${[...validSet].join(', ')}], got "${value}"`
    );
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function makeRequest(urlString, options, body) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(urlString);
    const isHttps = parsedUrl.protocol === 'https:';
    const transport = isHttps ? https : http;

    const reqOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: options.method || 'GET',
      headers: options.headers || {},
      timeout: options.timeoutMs || DEFAULT_TIMEOUT_MS,
    };

    const req = transport.request(reqOptions, (res) => {
      let rawData = '';
      res.on('data', chunk => { rawData += chunk; });
      res.on('end', () => {
        resolve({ statusCode: res.statusCode, headers: res.headers, body: rawData });
      });
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new BuyWhereSGError(`Request timed out after ${options.timeoutMs || DEFAULT_TIMEOUT_MS}ms`));
    });

    req.on('error', (err) => {
      reject(new BuyWhereSGError(`Network error: ${err.message}`));
    });

    if (body) {
      req.write(body);
    }

    req.end();
  });
}

async function executeWithRetry(requestFn, maxRetries, methodName) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (err) {
      if (err instanceof RateLimitError) {
        const waitMs = (err.retryAfterSeconds || 1) * 1000;
        if (attempt < maxRetries) {
          await sleep(waitMs);
          lastError = err;
          continue;
        }
        throw err;
      }
      if (err instanceof AuthenticationError || err instanceof ValidationError) {
        throw err;
      }
      if (err instanceof BuyWhereSGError && err.statusCode && err.statusCode < 500) {
        throw err;
      }
      lastError = err;
      if (attempt < maxRetries) {
        await sleep(RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt));
      }
    }
  }
  throw lastError;
}

class BuyWhereSGClient {
  constructor(options) {
    if (!options || typeof options !== 'object') {
      throw new AuthenticationError(
        'BuyWhereSGClient requires an options object with an apiKey field'
      );
    }
    if (!options.apiKey || typeof options.apiKey !== 'string' || options.apiKey.trim().length === 0) {
      throw new AuthenticationError(
        'BuyWhereSGClient requires a non-empty apiKey. Obtain one at https://buywhere-sg.nexus.io/keys'
      );
    }

    this._apiKey = options.apiKey.trim();
    this._baseUrl = (options.baseUrl || BUYWHERE_API_BASE).replace(/\/$/, '');
    this._timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
    this._maxRetries = options.maxRetries !== undefined ? options.maxRetries : DEFAULT_MAX_RETRIES;

    validatePositiveInteger(this._timeoutMs, 'timeoutMs', 1000, 120000);
    validatePositiveInteger(this._maxRetries, 'maxRetries', 0, 10);
  }

  _buildHeaders(extraHeaders) {
    return Object.assign(
      {
        'X-Api-Key': this._apiKey,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'buywhere-sg-sdk-js/1.0.0',
      },
      extraHeaders || {}
    );
  }

  _buildUrl(path, queryParams) {
    const url = new URL(this._baseUrl + path);
    if (queryParams && typeof queryParams === 'object') {
      for (const [key, value] of Object.entries(queryParams)) {
        if (value !== null && value !== undefined) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    return url.toString();
  }

  async _request(method, path, queryParams, body) {
    const urlString = this._buildUrl(path, queryParams);
    const headers = this._buildHeaders();
    const serializedBody = body ? JSON.stringify(body) : undefined;

    if (serializedBody) {
      headers['Content-Length'] = Buffer.byteLength(serializedBody);
    }

    const doRequest = async () => {
      const response = await makeRequest(
        urlString,
        { method, headers, timeoutMs: this._timeoutMs },
        serializedBody
      );

      if (response.statusCode === 401 || response.statusCode === 403) {
        throw new AuthenticationError(
          'Invalid or expired API key. Check your credentials at https://buywhere-sg.nexus.io/keys'
        );
      }

      if (response.statusCode === 429) {
        const retryAfter = parseInt(response.headers['retry-after'] || '60', 10);
        throw new RateLimitError(retryAfter);
      }

      let parsed;
      try {
        parsed = JSON.parse(response.body);
      } catch (_) {
        throw new BuyWhereSGError(
          `Non-JSON response from API (status ${response.statusCode}): ${response.body.slice(0, 200)}`,
          response.statusCode,
          response.body
        );
      }

      if (response.statusCode === 400) {
        throw new ValidationError(
          parsed.detail || parsed.message || 'Validation error from API'
        );
      }

      if (response.statusCode >= 400) {
        throw new BuyWhereSGError(
          parsed.detail || parsed.message || `API error ${response.statusCode}`,
          response.statusCode,
          parsed
        );
      }

      return parsed;
    };

    return executeWithRetry(doRequest, this._maxRetries, `${method} ${path}`);
  }

  /**
   * mainMethod — entry point para agentes: resuelve precios multi-vendor con dispersion analysis.
   * Alias de searchProductPrices para compatibilidad con el contrato `client.mainMethod(data)`.
   *
   * @param {object} data
   * @param {string} data.query - Texto de búsqueda del producto (requerido)
   * @param {string} [data.domain] - Dominio: 'electronics' | 'appliances' | 'computing'
   * @param {string} [data.vendor] - Vendor: 'lazada' | 'shopee' | 'courts' | 'harvey_norman' | 'all'
   * @param {string} [data.availability] - 'online' | 'physical' | 'both'
   * @param {string} [data.sort_by] - 'price_asc' | 'price_desc' | 'dispersion_desc' | 'freshness_desc'
   * @param {number} [data.limit] - Resultados a retornar (1-50, default 10)
   * @param {number} [data.page] - Página de resultados (1+, default 1)
   * @returns {Promise<PriceSearchResult>}
   */
  async mainMethod(data) {
    return this.searchProductPrices(data);
  }

  /**
   * searchProductPrices — busca precios multi-vendor con schema unificado SGD y Shannon dispersion.
   *
   * Cuando usarla: cuando un agente necesita comparar precios de un producto entre retailers SG
   * con metadatos de disponibilidad física u online.
   *
   * Cuando NO usarla: para obtener historial de precios de un SKU específico ya identificado
   * (usar fetchPriceHistory) o para analizar anomalías sobre un conjunto de SKUs (usar detectPriceAnomalies).
   *
   * @param {object} params
   * @returns {Promise<PriceSearchResult>}
   */
  async searchProductPrices(params) {
    if (params === null || params === undefined) {
      throw new ValidationError(
        'searchProductPrices requires a params object with at least a "query" field'
      );
    }
    if (typeof params !== 'object' || Array.isArray(params)) {
      throw new ValidationError(
        `searchProductPrices expects an object, got ${Array.isArray(params) ? 'array' : typeof params}`
      );
    }

    validateNonEmptyString(params.query, 'query');
    validateEnum(params.domain, 'domain', VALID_PRODUCT_DOMAINS);
    validateEnum(params.vendor, 'vendor', VALID_VENDORS);
    validateEnum(params.availability, 'availability', VALID_AVAILABILITY);
    validateEnum(params.sort_by, 'sort_by', VALID_SORT_BY);
    validatePositiveInteger(params.limit, 'limit', 1, 50);
    validatePositiveInteger(params.page, 'page', 1, 1000);

    const queryParams = {
      q: params.query.trim(),
      domain: params.domain || undefined,
      vendor: params.vendor || undefined,
      availability: params.availability || undefined,
      sort_by: params.sort_by || undefined,
      limit: params.limit || 10,
      page: params.page || 1,
    };

    return this._request('GET', '/prices', queryParams, null);
  }

  /**
   * fetchPriceHistory — retorna el historial de variación de precio en SGD para un SKU específico.
   *
   * Cuando usarla: cuando ya tienes un sku_id (obtenido de searchProductPrices) y necesitas
   * la serie temporal de precios para análisis de tendencia o validación de deal.
   *
   * Cuando NO usarla: para descubrimiento de productos — ese es el rol de searchProductPrices.
   * No admite IDs de producto externos (Lazada/Shopee) directamente.
   *
   * @param {string} skuId - SKU unificado BuyWhere (obtenido de searchProductPrices)
   * @param {object} [options]
   * @param {number} [options.days] - Ventana de historial en días (1-365, default 30)
   * @param {string} [options.vendor] - Filtrar historial por vendor específico
   * @param {string} [options.granularity] - 'hourly' | 'daily' | 'weekly' (default 'daily')
   * @returns {Promise<PriceHistoryResult>}
   */
  async fetchPriceHistory(skuId, options) {
    validateNonEmptyString(skuId, 'skuId');

    const opts = options || {};
    if (typeof opts !== 'object' || Array.isArray(opts)) {
      throw new ValidationError('options must be an object');
    }

    validatePositiveInteger(opts.days, 'days', 1, 365);
    validateEnum(opts.vendor, 'vendor', VALID_VENDORS);
    validateEnum(opts.granularity, 'granularity', new Set(['hourly', 'daily', 'weekly']));

    const queryParams = {
      days: opts.days || 30,
      vendor: opts.vendor || undefined,
      granularity: opts.granularity || 'daily',
    };

    return this._request('GET', `/prices/${encodeURIComponent(skuId.trim())}/history`, queryParams, null);
  }

  /**
   * detectPriceAnomalies — aplica análisis de entropía de Shannon sobre distribución de precios
   * multi-vendor para identificar SKUs con dispersión anómala en tiempo real.
   *
   * Cuando usarla: cuando un agente necesita detectar si un precio específico es outlier
   * respecto al mercado SG, o para auditar un conjunto de SKUs buscando manipulación de precio.
   *
   * Cuando NO usarla: para búsqueda de productos (usar searchProductPrices) o para historial
   * de un solo SKU (usar fetchPriceHistory). Requiere entre 2 y 100 SKUs por llamada.
   *
   * @param {string[]} skuIds - Array de SKU IDs unificados BuyWhere (2-100 elementos)
   * @param {object} [options]
   * @param {number} [options.dispersion_threshold_bits] - Umbral mínimo de bits para reportar anomalía (default 1.5)
   * @param {string} [options.domain] - Filtrar por dominio de producto
   * @returns {Promise<AnomalyDetectionResult>}
   */
  async detectPriceAnomalies(skuIds, options) {
    if (skuIds === null || skuIds === undefined) {
      throw new ValidationError('skuIds is required and cannot be null or undefined');
    }
    if (!Array.isArray(skuIds)) {
      throw new ValidationError(`skuIds must be an array, got ${typeof skuIds}`);
    }
    if (skuIds.length < 2) {
      throw new ValidationError(
        `skuIds must contain at least 2 SKUs for dispersion analysis, got ${skuIds.length}`
      );
    }
    if (skuIds.length > 100) {
      throw new ValidationError(
        `skuIds cannot exceed 100 elements per call, got ${skuIds.length}. Split into batches.`
      );
    }

    for (let i = 0; i < skuIds.length; i++) {
      if (typeof skuIds[i] !== 'string' || skuIds[i].trim().length === 0) {
        throw new ValidationError(
          `skuIds[${i}] must be a non-empty string, got ${JSON.stringify(skuIds[i])}`
        );
      }
    }

    const opts = options || {};
    if (typeof opts !== 'object' || Array.isArray(opts)) {
      throw new ValidationError('options must be an object');
    }

    if (opts.dispersion_threshold_bits !== undefined && opts.dispersion_threshold_bits !== null) {
      if (typeof opts.dispersion_threshold_bits !== 'number' || opts.dispersion_threshold_bits < 0) {
        throw new ValidationError('dispersion_threshold_bits must be a non-negative number');
      }
    }

    validateEnum(opts.domain, 'domain', VALID_PRODUCT_DOMAINS);

    const body = {
      sku_ids: skuIds.map(id => id.trim()),
      dispersion_threshold_bits: opts.dispersion_threshold_bits !== undefined
        ? opts.dispersion_threshold_bits
        : 1.5,
      domain: opts.domain || undefined,
    };

    return this._request('POST', '/prices/anomalies', null, body);
  }

  /**
   * resolveVendorAvailability — verifica disponibilidad en tiempo real (stock + tienda física)
   * para un SKU en un vendor específico sin disparar un re-scrape completo del catálogo.
   *
   * Cuando usarla: paso final antes de recomendar un producto a un usuario — confirma stock
   * y ubicación de tienda física si availability='physical' o 'both'.
   *
   * Cuando NO usarla: para comparación de precios entre múltiples vendors (usar searchProductPrices)
   * o si solo necesitas el precio sin importar el stock actual.
   *
   * @param {string} skuId - SKU unificado BuyWhere
   * @param {string} vendor - Vendor a consultar ('lazada' | 'shopee' | 'courts' | 'harvey_norman')
   * @param {object} [options]
   * @param {string} [options.postal_code] - Código postal SG de 6 dígitos para proximidad de tienda
   * @returns {Promise<VendorAvailabilityResult>}
   */
  async resolveVendorAvailability(skuId, vendor, options) {
    validateNonEmptyString(skuId, 'skuId');
    validateNonEmptyString(vendor, 'vendor');

    const vendorWithoutAll = new Set(['lazada', 'shopee', 'courts', 'harvey_norman']);
    validateEnum(vendor, 'vendor', vendorWithoutAll);

    const opts = options || {};
    if (typeof opts !== 'object' || Array.isArray(opts)) {
      throw new ValidationError('options must be an object');
    }

    if (opts.postal_code !== undefined && opts.postal_code !== null) {
      if (typeof opts.postal_code !== 'string') {
        throw new ValidationError('postal_code must be a string');
      }
      if (!/^\d{6}$/.test(opts.postal_code)) {
        throw new ValidationError(
          `postal_code must be a 6-digit Singapore postal code, got "${opts.postal_code}"`
        );
      }
    }

    const queryParams = {
      vendor: vendor.trim(),
      postal_code: opts.postal_code || undefined,
    };

    return this._request(
      'GET',
      `/prices/${encodeURIComponent(skuId.trim())}/availability`,
      queryParams,
      null
    );
  }
}

module.exports = BuyWhereSGClient;
module.exports.BuyWhereSGClient = BuyWhereSGClient;
module.exports.BuyWhereSGError = BuyWhereSGError;
module.exports.AuthenticationError = AuthenticationError;
module.exports.RateLimitError = RateLimitError;
module.exports.ValidationError = ValidationError;
```