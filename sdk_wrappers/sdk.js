
```javascript
'use strict';

const https = require('https');
const http = require('http');
const { URL } = require('url');

const BUYWHERE_API_BASE = process.env.BUYWHERE_API_BASE || 'https://api.buywhere.sg/v1';
const DEFAULT_TIMEOUT_MS = 15000;
const DEFAULT_MAX_RESULTS = 20;
const MIN_QUERY_LENGTH = 2;
const MAX_QUERY_LENGTH = 512;
const VALID_SORT_FIELDS = ['value_score', 'price_sgd', 'reliability_score', 'vendor_count'];
const VALID_SORT_ORDERS = ['asc', 'desc'];

class BuyWhereAuthError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BuyWhereAuthError';
    this.statusCode = 401;
  }
}

class BuyWhereValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'BuyWhereValidationError';
    this.field = field || null;
    this.statusCode = 400;
  }
}

class BuyWhereRateLimitError extends Error {
  constructor(retryAfterSeconds) {
    super(`Rate limit exceeded. Retry after ${retryAfterSeconds}s`);
    this.name = 'BuyWhereRateLimitError';
    this.retryAfterSeconds = retryAfterSeconds;
    this.statusCode = 429;
  }
}

class BuyWhereAPIError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = 'BuyWhereAPIError';
    this.statusCode = statusCode;
    this.responseBody = body || null;
  }
}

function validateApiKey(apiKey) {
  if (apiKey === null || apiKey === undefined) {
    throw new BuyWhereAuthError(
      'API key is required. Pass it via options.apiKey or set BUYWHERE_API_KEY environment variable.'
    );
  }
  if (typeof apiKey !== 'string') {
    throw new BuyWhereAuthError(
      `API key must be a string, received ${typeof apiKey}.`
    );
  }
  if (apiKey.trim().length === 0) {
    throw new BuyWhereAuthError(
      'API key must not be an empty string.'
    );
  }
}

function validateSearchQuery(query) {
  if (query === null || query === undefined) {
    throw new BuyWhereValidationError(
      'Search query is required and must not be null or undefined.',
      'query'
    );
  }
  if (typeof query !== 'string') {
    throw new BuyWhereValidationError(
      `Search query must be a string, received ${typeof query}.`,
      'query'
    );
  }
  const trimmed = query.trim();
  if (trimmed.length < MIN_QUERY_LENGTH) {
    throw new BuyWhereValidationError(
      `Search query must be at least ${MIN_QUERY_LENGTH} characters long.`,
      'query'
    );
  }
  if (trimmed.length > MAX_QUERY_LENGTH) {
    throw new BuyWhereValidationError(
      `Search query must not exceed ${MAX_QUERY_LENGTH} characters.`,
      'query'
    );
  }
  return trimmed;
}

function validateProductId(productId) {
  if (productId === null || productId === undefined) {
    throw new BuyWhereValidationError(
      'Product ID is required and must not be null or undefined.',
      'productId'
    );
  }
  if (typeof productId !== 'string' && typeof productId !== 'number') {
    throw new BuyWhereValidationError(
      `Product ID must be a string or number, received ${typeof productId}.`,
      'productId'
    );
  }
  const str = String(productId).trim();
  if (str.length === 0) {
    throw new BuyWhereValidationError(
      'Product ID must not be empty.',
      'productId'
    );
  }
  return str;
}

function validateSearchOptions(options) {
  const validated = {};

  if (options.maxResults !== undefined) {
    const n = Number(options.maxResults);
    if (!Number.isInteger(n) || n < 1 || n > 100) {
      throw new BuyWhereValidationError(
        'maxResults must be an integer between 1 and 100.',
        'maxResults'
      );
    }
    validated.max_results = n;
  } else {
    validated.max_results = DEFAULT_MAX_RESULTS;
  }

  if (options.minPriceSGD !== undefined) {
    const v = Number(options.minPriceSGD);
    if (isNaN(v) || v < 0) {
      throw new BuyWhereValidationError(
        'minPriceSGD must be a non-negative number.',
        'minPriceSGD'
      );
    }
    validated.min_price_sgd = v;
  }

  if (options.maxPriceSGD !== undefined) {
    const v = Number(options.maxPriceSGD);
    if (isNaN(v) || v < 0) {
      throw new BuyWhereValidationError(
        'maxPriceSGD must be a non-negative number.',
        'maxPriceSGD'
      );
    }
    validated.max_price_sgd = v;
  }

  if (
    validated.min_price_sgd !== undefined &&
    validated.max_price_sgd !== undefined &&
    validated.min_price_sgd > validated.max_price_sgd
  ) {
    throw new BuyWhereValidationError(
      'minPriceSGD must not be greater than maxPriceSGD.',
      'minPriceSGD'
    );
  }

  if (options.sortBy !== undefined) {
    if (!VALID_SORT_FIELDS.includes(options.sortBy)) {
      throw new BuyWhereValidationError(
        `sortBy must be one of: ${VALID_SORT_FIELDS.join(', ')}.`,
        'sortBy'
      );
    }
    validated.sort_by = options.sortBy;
  }

  if (options.sortOrder !== undefined) {
    if (!VALID_SORT_ORDERS.includes(options.sortOrder)) {
      throw new BuyWhereValidationError(
        `sortOrder must be one of: ${VALID_SORT_ORDERS.join(', ')}.`,
        'sortOrder'
      );
    }
    validated.sort_order = options.sortOrder;
  }

  if (options.categories !== undefined) {
    if (!Array.isArray(options.categories)) {
      throw new BuyWhereValidationError(
        'categories must be an array of strings.',
        'categories'
      );
    }
    for (const cat of options.categories) {
      if (typeof cat !== 'string' || cat.trim().length === 0) {
        throw new BuyWhereValidationError(
          'Each element in categories must be a non-empty string.',
          'categories'
        );
      }
    }
    validated.categories = options.categories.map(c => c.trim());
  }

  if (options.minValueScore !== undefined) {
    const v = Number(options.minValueScore);
    if (isNaN(v) || v < 0 || v > 1) {
      throw new BuyWhereValidationError(
        'minValueScore must be a number between 0.0 and 1.0.',
        'minValueScore'
      );
    }
    validated.min_value_score = v;
  }

  return validated;
}

function makeHttpRequest(url, options, timeoutMs) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const isHttps = parsedUrl.protocol === 'https:';
    const lib = isHttps ? https : http;

    const requestOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: options.method || 'GET',
      headers: options.headers || {},
    };

    const req = lib.request(requestOptions, (res) => {
      let rawData = '';
      res.on('data', (chunk) => { rawData += chunk; });
      res.on('end', () => {
        resolve({ statusCode: res.statusCode, headers: res.headers, body: rawData });
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new BuyWhereAPIError(
        `Request timed out after ${timeoutMs}ms`,
        408,
        null
      ));
    });

    req.on('error', (err) => {
      reject(new BuyWhereAPIError(
        `Network error: ${err.message}`,
        0,
        null
      ));
    });

    if (options.body) {
      req.write(options.body);
    }

    req.end();
  });
}

function handleHttpResponse(response) {
  const { statusCode, headers, body } = response;

  let parsed = null;
  try {
    parsed = JSON.parse(body);
  } catch (_) {
    if (statusCode < 200 || statusCode >= 300) {
      throw new BuyWhereAPIError(
        `API returned non-JSON error response with status ${statusCode}`,
        statusCode,
        body
      );
    }
  }

  if (statusCode === 401) {
    throw new BuyWhereAuthError(
      (parsed && parsed.detail) || 'Invalid or expired API key.'
    );
  }

  if (statusCode === 429) {
    const retryAfter = parseInt(headers['retry-after'] || '60', 10);
    throw new BuyWhereRateLimitError(retryAfter);
  }

  if (statusCode === 400) {
    throw new BuyWhereValidationError(
      (parsed && parsed.detail) || `Bad request: ${body}`,
      null
    );
  }

  if (statusCode < 200 || statusCode >= 300) {
    throw new BuyWhereAPIError(
      (parsed && parsed.detail) || `API error with status ${statusCode}`,
      statusCode,
      body
    );
  }

  return parsed;
}

class BuyWhereSingaporeClient {
  constructor(options) {
    if (options === null || options === undefined) {
      options = {};
    }
    if (typeof options !== 'object' || Array.isArray(options)) {
      throw new BuyWhereValidationError(
        'Client options must be a plain object.',
        'options'
      );
    }

    const apiKey = options.apiKey || process.env.BUYWHERE_API_KEY;
    validateApiKey(apiKey);

    this._apiKey = apiKey;
    this._baseUrl = (options.baseUrl || BUYWHERE_API_BASE).replace(/\/$/, '');
    this._timeoutMs = options.timeoutMs !== undefined
      ? Number(options.timeoutMs)
      : DEFAULT_TIMEOUT_MS;

    if (isNaN(this._timeoutMs) || this._timeoutMs < 500 || this._timeoutMs > 60000) {
      throw new BuyWhereValidationError(
        'timeoutMs must be a number between 500 and 60000.',
        'timeoutMs'
      );
    }
  }

  _buildHeaders() {
    return {
      'Authorization': `Bearer ${this._apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client': 'buywhere-sg-sdk-node/1.0.0',
    };
  }

  async _get(path, queryParams) {
    const url = new URL(this._baseUrl + path);
    if (queryParams) {
      for (const [key, value] of Object.entries(queryParams)) {
        if (value !== undefined && value !== null) {
          if (Array.isArray(value)) {
            for (const v of value) {
              url.searchParams.append(key, String(v));
            }
          } else {
            url.searchParams.set(key, String(value));
          }
        }
      }
    }

    const response = await makeHttpRequest(
      url.toString(),
      { method: 'GET', headers: this._buildHeaders() },
      this._timeoutMs
    );

    return handleHttpResponse(response);
  }

  async _post(path, bodyObject) {
    const url = this._baseUrl + path;
    const bodyStr = JSON.stringify(bodyObject);

    const response = await makeHttpRequest(
      url,
      {
        method: 'POST',
        headers: {
          ...this._buildHeaders(),
          'Content-Length': Buffer.byteLength(bodyStr),
        },
        body: bodyStr,
      },
      this._timeoutMs
    );

    return handleHttpResponse(response);
  }

  async searchProductsBySemanticQuery(query, options) {
    const cleanQuery = validateSearchQuery(query);
    const opts = options !== undefined ? options : {};

    if (typeof opts !== 'object' || Array.isArray(opts)) {
      throw new BuyWhereValidationError(
        'options must be a plain object.',
        'options'
      );
    }

    const validatedOpts = validateSearchOptions(opts);

    const payload = {
      query: cleanQuery,
      ...validatedOpts,
    };

    return this._post('/products/semantic-search', payload);
  }

  async getProductValueScore(productId) {
    const cleanId = validateProductId(productId);
    return this._get(`/products/${encodeURIComponent(cleanId)}/value-score`);
  }

  async getVendorPriceDistribution(productId) {
    const cleanId = validateProductId(productId);
    return this._get(`/products/${encodeURIComponent(cleanId)}/vendor-prices`);
  }

  async rankProductsByValueScore(productIds, options) {
    if (productIds === null || productIds === undefined) {
      throw new BuyWhereValidationError(
        'productIds is required and must not be null or undefined.',
        'productIds'
      );
    }
    if (!Array.isArray(productIds)) {
      throw new BuyWhereValidationError(
        `productIds must be an array, received ${typeof productIds}.`,
        'productIds'
      );
    }
    if (productIds.length === 0) {
      throw new BuyWhereValidationError(
        'productIds must contain at least one product ID.',
        'productIds'
      );
    }
    if (productIds.length > 50) {
      throw new BuyWhereValidationError(
        'productIds must not contain more than 50 product IDs per request.',
        'productIds'
      );
    }

    const cleanIds = productIds.map((id, index) => {
      if (id === null || id === undefined) {
        throw new BuyWhereValidationError(
          `productIds[${index}] must not be null or undefined.`,
          'productIds'
        );
      }
      if (typeof id !== 'string' && typeof id !== 'number') {
        throw new BuyWhereValidationError(
          `productIds[${index}] must be a string or number, received ${typeof id}.`,
          'productIds'
        );
      }
      const str = String(id).trim();
      if (str.length === 0) {
        throw new BuyWhereValidationError(
          `productIds[${index}] must not be an empty string.`,
          'productIds'
        );
      }
      return str;
    });

    const opts = options !== undefined ? options : {};
    if (typeof opts !== 'object' || Array.isArray(opts)) {
      throw new BuyWhereValidationError(
        'options must be a plain object.',
        'options'
      );
    }

    const payload = {
      product_ids: cleanIds,
    };

    if (opts.penalizeIntraSessionVariance !== undefined) {
      if (typeof opts.penalizeIntraSessionVariance !== 'boolean') {
        throw new BuyWhereValidationError(
          'penalizeIntraSessionVariance must be a boolean.',
          'penalizeIntraSessionVariance'
        );
      }
      payload.penalize_intra_session_variance = opts.penalizeIntraSessionVariance;
    }

    if (opts.entropyWeight !== undefined) {
      const v = Number(opts.entropyWeight);
      if (isNaN(v) || v < 0 || v > 1) {
        throw new BuyWhereValidationError(
          'entropyWeight must be a number between 0.0 and 1.0.',
          'entropyWeight'
        );
      }
      payload.entropy_weight = v;
    }

    return this._post('/products/rank-by-value-score', payload);
  }

  async mainMethod(data) {
    if (data === null || data === undefined) {
      throw new BuyWhereValidationError(
        'data is required. Pass an object with at least a "query" field for semantic search, ' +
        'or a "productId" field for value-score lookup.',
        'data'
      );
    }
    if (typeof data !== 'object' || Array.isArray(data)) {
      throw new BuyWhereValidationError(
        `data must be a plain object, received ${Array.isArray(data) ? 'array' : typeof data}.`,
        'data'
      );
    }

    const hasQuery = data.query !== undefined && data.query !== null;
    const hasProductId = data.productId !== undefined && data.productId !== null;
    const hasProductIds = data.productIds !== undefined && data.productIds !== null;

    if (!hasQuery && !hasProductId && !hasProductIds) {
      throw new BuyWhereValidationError(
        'data must contain at least one of: "query" (string for semantic search), ' +
        '"productId" (string/number for value-score lookup), or ' +
        '"productIds" (array for batch ranking).',
        'data'
      );
    }

    if (hasQuery) {
      const { query, ...searchOptions } = data;
      return this.searchProductsBySemanticQuery(query, searchOptions);
    }

    if (hasProductIds) {
      const { productIds, ...rankOptions } = data;
      return this.rankProductsByValueScore(productIds, rankOptions);
    }

    return this.getProductValueScore(data.productId);
  }
}

function createBuyWhereClient(options) {
  return new BuyWhereSingaporeClient(options || {});
}

const _defaultClientCache = { instance: null, key: null };

function _getOrCreateDefaultClient() {
  const currentKey = process.env.BUYWHERE_API_KEY;
  if (!currentKey) {
    throw new BuyWhereAuthError(
      'No API key found. Set BUYWHERE_API_KEY environment variable or use createBuyWhereClient({ apiKey: "..." }).'
    );
  }
  if (_defaultClientCache.instance === null || _defaultClientCache.key !== currentKey) {
    _defaultClientCache.instance = new BuyWhereSingaporeClient({ apiKey: currentKey });
    _defaultClientCache.key = currentKey;
  }
  return _defaultClientCache.instance;
}

const moduleExports = {
  createBuyWhereClient,
  BuyWhereSingaporeClient,
  BuyWhereAuthError,
  BuyWhereValidationError,
  BuyWhereRateLimitError,
  BuyWhereAPIError,

  async mainMethod(data) {
    return _getOrCreateDefaultClient().mainMethod(data);
  },

  async searchProductsBySemanticQuery(query, options) {
    return _getOrCreateDefaultClient().searchProductsBySemanticQuery(query, options);
  },

  async getProductValueScore(productId) {
    return _getOrCreateDefaultClient().getProductValueScore(productId);
  },

  async getVendorPriceDistribution(productId) {
    return _getOrCreateDefaultClient().getVendorPriceDistribution(productId);
  },

  async rankProductsByValueScore(productIds, options) {
    return _getOrCreateDefaultClient().rankProductsByValueScore(productIds, options);
  },
};

module.exports = moduleExports;
```