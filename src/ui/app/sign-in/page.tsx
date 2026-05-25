import { Suspense } from "react";
import { SignInForm } from "@/components/auth/sign-in-form";

export default function SignInPage() {
  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-between bg-primary p-10 text-primary-foreground">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-md bg-primary-foreground/10 flex items-center justify-center text-lg font-bold">
            R
          </div>
          <span className="text-lg font-semibold">ReID Production</span>
        </div>
        <div className="space-y-4">
          <p className="text-2xl font-medium leading-snug">
            Real-time person re-identification across your camera network.
          </p>
          <p className="text-sm text-primary-foreground/70">
            Track persons across devices, audit sightings, and run structured
            searches over your detection pipeline.
          </p>
        </div>
      </div>

      <div className="flex items-center justify-center p-6 sm:p-10">
        <div className="mx-auto w-full max-w-sm space-y-6">
          <div className="space-y-2 text-center sm:text-left">
            <h1 className="text-2xl font-semibold tracking-tight">Sign in</h1>
            <p className="text-sm text-muted-foreground">
              Use your gateway credentials to access the console.
            </p>
          </div>
          <Suspense>
            <SignInForm />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
