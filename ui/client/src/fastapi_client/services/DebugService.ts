/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ConfigResponse } from '../models/ConfigResponse';
import type { TestConnectionResponse } from '../models/TestConnectionResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DebugService {
    /**
     * Get Debug Config
     * Get current configuration for debugging (no secrets exposed).
     * @returns ConfigResponse Successful Response
     * @throws ApiError
     */
    public static getDebugConfigApiDebugConfigGet(): CancelablePromise<ConfigResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/debug/config',
        });
    }
    /**
     * Test Workspace Connection
     * Test workspace connection and authentication.
     * @returns TestConnectionResponse Successful Response
     * @throws ApiError
     */
    public static testWorkspaceConnectionApiDebugTestWorkspacePost(): CancelablePromise<TestConnectionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/debug/test-workspace',
        });
    }
    /**
     * Test Agent Connection
     * Test agent endpoint reachability (without sending actual message).
     * @returns TestConnectionResponse Successful Response
     * @throws ApiError
     */
    public static testAgentConnectionApiDebugTestAgentPost(): CancelablePromise<TestConnectionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/debug/test-agent',
        });
    }
    /**
     * Get Recent Logs
     * Get recent application logs for debugging.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getRecentLogsApiDebugLogsGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/debug/logs',
        });
    }
}
