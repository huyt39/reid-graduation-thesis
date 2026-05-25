import { z } from "zod";

export const signInSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

export type SignInFormData = z.infer<typeof signInSchema>;

export type Role = "admin" | "operator" | "viewer";

/** The reid gateway carries identity in the JWT payload — no /auth/me endpoint. */
export interface User {
  username: string;
  role: Role;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  rememberMe: boolean;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface AuthActions {
  setUser: (user: User | null) => void;
  setAccessToken: (token: string | null) => void;
  setRememberMe: (remember: boolean) => void;
  setIsLoading: (loading: boolean) => void;
  login: (accessToken: string, user: User, remember?: boolean) => void;
  logout: () => void;
  clearTokens: () => void;
}

export type AuthStore = AuthState & AuthActions;
