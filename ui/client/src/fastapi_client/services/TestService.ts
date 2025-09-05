/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TableTestResponse } from '../models/TableTestResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class TestService {
    /**
     * Test Table Stream
     * Stream table content in chunks to test rendering.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static testTableStreamApiApiTestTableStreamGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/test/table-stream',
        });
    }
    /**
     * Get Good Table
     * Return properly formatted table content for testing
     * @returns TableTestResponse Successful Response
     * @throws ApiError
     */
    public static getGoodTableApiApiTestTableGoodGet(): CancelablePromise<TableTestResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/test/table/good',
        });
    }
    /**
     * Get Malformed Table
     * Return malformed table content that mimics agent output
     * @returns TableTestResponse Successful Response
     * @throws ApiError
     */
    public static getMalformedTableApiApiTestTableMalformedGet(): CancelablePromise<TableTestResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/test/table/malformed',
        });
    }
    /**
     * Validate Table Content
     * Validate table content and return analysis
     * @param content
     * @returns any Successful Response
     * @throws ApiError
     */
    public static validateTableContentApiApiTestTableValidatePost(
        content: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/api/test/table/validate',
            query: {
                'content': content,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Test Health
     * Simple health check for test endpoints
     * @returns any Successful Response
     * @throws ApiError
     */
    public static testHealthApiApiTestHealthGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/test/health',
        });
    }
}
