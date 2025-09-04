/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ChatRequest } from '../models/ChatRequest';
import type { ChatResponse } from '../models/ChatResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ChatService {
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
}
