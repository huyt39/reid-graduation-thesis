/**
 * Auth Store (Zustand)
 *
 * Manages authentication state in the React app. The reid gateway issues a
 * single access token containing username + role in its payload, so the user
 * is derived directly from the token rather than fetched from /auth/me.
 */
"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { AuthStore, User } from "./types";
import {
  saveToken,
  loadToken,
  clearToken,
  loadRememberMe,
  REMEMBER_ME_KEY,
} from "./token-storage";

export const AUTH_STORE_KEY = "reid-auth";

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => {
      const { accessToken } = loadToken();

      return {
        user: null,
        accessToken,
        rememberMe: loadRememberMe(),
        isAuthenticated: !!accessToken,
        isLoading: false,

        setUser: (user: User | null) => set({ user }),

        setAccessToken: (token: string | null) => {
          const { rememberMe } = get();
          saveToken(token, rememberMe);
          set({
            accessToken: token,
            isAuthenticated: !!token,
          });
        },

        setRememberMe: (remember: boolean) => {
          if (typeof window !== "undefined") {
            localStorage.setItem(REMEMBER_ME_KEY, String(remember));
          }
          set({ rememberMe: remember });
        },

        setIsLoading: (loading: boolean) => set({ isLoading: loading }),

        login: (accessToken, user, remember) => {
          const { rememberMe } = get();
          const useRemember = remember ?? rememberMe;
          saveToken(accessToken, useRemember);
          set({
            user,
            accessToken,
            isAuthenticated: true,
            isLoading: false,
            rememberMe: useRemember,
          });
        },

        logout: () => {
          clearToken();
          set({
            user: null,
            accessToken: null,
            isAuthenticated: false,
            isLoading: false,
          });
        },

        clearTokens: () => {
          clearToken();
          set({
            accessToken: null,
            isAuthenticated: false,
          });
        },
      };
    },
    {
      name: AUTH_STORE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        rememberMe: state.rememberMe,
      }),
    }
  )
);

export const useUser = () => useAuthStore((state) => state.user);
export const useIsAuthenticated = () => useAuthStore((state) => state.isAuthenticated);
export const useAuthLoading = () => useAuthStore((state) => state.isLoading);
export const useAccessToken = () => useAuthStore((state) => state.accessToken);
