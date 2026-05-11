"use client";

import { HTMLAttributes, useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { LoaderCircle } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/lib/auth/auth-store";
import { authClient, userFromToken } from "@/lib/auth/auth-client";
import { saveToken } from "@/lib/auth/token-storage";
import { signInSchema, type SignInFormData } from "@/lib/auth/types";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { PasswordInput } from "@/components/ui/password-input";

type SignInFormProps = HTMLAttributes<HTMLFormElement>;

export function SignInForm({ className, ...props }: SignInFormProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();

  const { login, rememberMe, setRememberMe } = useAuthStore();

  useEffect(() => {
    setMounted(true);
  }, []);

  const form = useForm<SignInFormData>({
    resolver: zodResolver(signInSchema),
    defaultValues: { username: "", password: "" },
  });

  async function onSubmit(data: SignInFormData) {
    setIsLoading(true);
    try {
      const response = await authClient.signIn(data);
      if (response.error || !response.data) {
        toast.error(response.error || "Login failed");
        return;
      }

      const { access_token } = response.data;
      saveToken(access_token, rememberMe);

      const user = userFromToken(access_token);
      if (!user) {
        toast.error("Invalid token returned from server");
        return;
      }

      login(access_token, user, rememberMe);
      toast.success("Welcome back");

      const redirectUrl = searchParams.get("redirect") || "/";
      router.push(redirectUrl);
    } catch (error) {
      console.error("Login failed:", error);
      toast.error("An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className={cn("grid gap-3", className)}
        {...props}
      >
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Username</FormLabel>
              <FormControl>
                <Input placeholder="admin" autoComplete="username" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem className="relative">
              <FormLabel>Password</FormLabel>
              <FormControl>
                <PasswordInput
                  placeholder="********"
                  autoComplete="current-password"
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <div className="flex flex-col gap-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:gap-0">
          <div className="flex items-center gap-2">
            <Checkbox
              id="rememberMe"
              className="border-gray-400"
              checked={mounted ? rememberMe : false}
              onCheckedChange={(checked) => setRememberMe(Boolean(checked))}
            />
            <label
              htmlFor="rememberMe"
              className="cursor-pointer text-sm font-medium select-none"
            >
              Remember me
            </label>
          </div>
        </div>

        <Button className="mt-3 py-6" disabled={isLoading}>
          {isLoading ? (
            <div className="flex items-center gap-2">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              <span>Signing in...</span>
            </div>
          ) : (
            "Sign in"
          )}
        </Button>
      </form>
    </Form>
  );
}
