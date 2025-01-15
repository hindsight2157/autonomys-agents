import axios from 'axios'
import { useQuery } from 'react-query'
import type { AgentMemory, DSNResponse } from '../types'
import { ResponseStatus } from '../types/enums'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

export const useMemory = (cid: string) => {
    return useQuery<AgentMemory, Error>(
        ['memory', cid],
        async () => {
            const { data } = await api.get(`/memories/${cid}`)
            return data
        },
        {
            enabled: !!cid,
            retry: (failureCount, error) => {
                if (axios.isAxiosError(error) && error.response?.status === 404) {
                    return false;
                }
                return failureCount < 3;
            },
            retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
        }
    )
}

export const useLatestMemory = (agentId: string) => {
    return useQuery<AgentMemory, Error>(
        ['latestMemory', agentId],
        async () => {
            const { data } = await api.get(`/agents/${agentId}/memories/latest`)
            return data
        },
        {
            enabled: !!agentId,
        }
    )
}

export const useDSNData = (
    page: number = 1, 
    limit: number = 10, 
    type: ResponseStatus | 'all' = 'all',
    search?: string,
    author?: string
) => {
    return useQuery<DSNResponse, Error>(
        ['dsn', page, limit, type, search, author],
        async () => {
            const params = new URLSearchParams({
                page: page.toString(),
                limit: limit.toString(),
            });
            
            if (type !== 'all') {
                params.append('type', type);
            }

            if (search) {
                params.append('search', search);
            }

            if (author) {
                params.append('author', author);
            }
            
            const { data } = await api.get(`/memories?${params}`);
            return data;
        },
        {
            retry: 3,
            retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
            keepPreviousData: true,
        }
    );
};


