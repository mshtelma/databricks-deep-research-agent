/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ChatRequest } from '../models/ChatRequest';
import type { ChatResponse } from '../models/ChatResponse';
import type { ConfigResponse } from '../models/ConfigResponse';
import type { TableTestResponse } from '../models/TableTestResponse';
import type { TestConnectionResponse } from '../models/TestConnectionResponse';
import type { UserInfo } from '../models/UserInfo';
import type { UserWorkspaceInfo } from '../models/UserWorkspaceInfo';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ApiService {
    /**
     * Get Current User
     * Get current user information from Databricks.
     * @returns UserInfo Successful Response
     * @throws ApiError
     */
    public static getCurrentUserApiUserMeGet(): CancelablePromise<UserInfo> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/user/me',
        });
    }
    /**
     * Get User Workspace Info
     * Get user information along with workspace details.
     * @returns UserWorkspaceInfo Successful Response
     * @throws ApiError
     */
    public static getUserWorkspaceInfoApiUserMeWorkspaceGet(): CancelablePromise<UserWorkspaceInfo> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/user/me/workspace',
        });
    }
    /**
     * Send Message
     * Send message to agent (non-streaming response).
     * @param requestBody
     * @returns ChatResponse Successful Response
     * @throws ApiError
     */
    public static sendMessageApiChatSendPost(
        requestBody: ChatRequest,
    ): CancelablePromise<ChatResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/chat/send',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Stream Message
     * Stream message to agent (real-time responses).
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamMessageApiChatStreamPost(
        requestBody: ChatRequest,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/chat/stream',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Debug Progress Test
     * Test endpoint that emits sample progress events.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static debugProgressTestApiChatDebugProgressTestGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/chat/debug/progress-test',
        });
    }
    /**
     * Chat Health
     * Health check for chat service.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static chatHealthApiChatHealthGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/chat/health',
        });
    }
    /**
     * Get Chat Config
     * Get chat configuration options.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getChatConfigApiChatConfigGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/chat/config',
        });
    }
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
    /**
     * Get Table Test
     * Get test table content for markdown rendering debugging.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getTableTestApiDebugTableTestGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/debug/table-test',
        });
    }
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
