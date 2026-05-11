"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/lib/auth/auth-store";
import { authClient, userFromToken } from "@/lib/auth/auth-client";
import { Spinner } from "@/components/ui/spinner";

interface AuthGuardProps {
  children: React.ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, user, accessToken, setUser, logout, setIsLoading } = useAuthStore();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      if (isAuthenticated && !user && accessToken) {
        setIsLoading(true);
        const decoded = userFromToken(accessToken);

        if (decoded) {
          setUser(decoded);
        } else {
          // Token unparseable — try refresh
          const newToken = await authClient.refreshAccessToken();
          const newUser = newToken ? userFromToken(newToken) : null;
          if (newUser) {
            setUser(newUser);
          } else {
            logout();
            router.push(`/sign-in?redirect=${encodeURIComponent(pathname)}`);
          }
        }

        setIsLoading(false);
      } else if (!isAuthenticated) {
        router.push(`/sign-in?redirect=${encodeURIComponent(pathname)}`);
      }

      setIsChecking(false);
    };

    checkAuth();
  }, [isAuthenticated, user, accessToken, pathname, router, setUser, logout, setIsLoading]);

  if (isChecking || (isAuthenticated && !user)) {
    return (
      <div className="flex h-screen w-screen items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Spinner className="h-8 w-8" />
          <p className="text-muted-foreground text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return <>{children}</>;
}
