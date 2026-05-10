"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated()) return;
    router.replace("/dashboard");
  }, [router]);

  return (
    <div className="h-screen bg-surface" />
  );
}
