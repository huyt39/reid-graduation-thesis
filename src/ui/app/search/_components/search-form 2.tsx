"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useSearch } from "@/hooks/use-search";

const schema = z.object({
  query_type: z.string().min(1),
  gender: z.string().optional(),
  device_id: z.string().optional(),
  start_time: z.string().optional(),
  end_time: z.string().optional(),
});

type FormValues = z.infer<typeof schema>;

const QUERY_TYPES = ["person_count", "device_activity", "gender_distribution"];

export function SearchForm() {
  const [submitted, setSubmitted] = useState(false);
  const { result, isLoading, error, run } = useSearch();

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { query_type: QUERY_TYPES[0] },
  });

  async function onSubmit(values: FormValues) {
    setSubmitted(true);
    const { query_type, ...params } = values;
    const cleaned = Object.fromEntries(
      Object.entries(params).filter(([, v]) => v !== "" && v !== undefined)
    );
    await run(query_type, cleaned);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Structured query</CardTitle>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <FormField
                control={form.control}
                name="query_type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Query type</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {QUERY_TYPES.map((q) => (
                          <SelectItem key={q} value={q}>
                            {q}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="gender"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Gender (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="male / female" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="device_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Device ID (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="cam-01" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="grid grid-cols-2 gap-3">
                <FormField
                  control={form.control}
                  name="start_time"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Start</FormLabel>
                      <FormControl>
                        <Input type="datetime-local" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="end_time"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>End</FormLabel>
                      <FormControl>
                        <Input type="datetime-local" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <Button type="submit" disabled={isLoading} className="w-full">
                <Search className="h-4 w-4" />
                {isLoading ? "Running..." : "Run query"}
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Result</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : error ? (
            <p className="text-destructive text-sm">{error}</p>
          ) : !submitted ? (
            <p className="text-sm text-muted-foreground">
              Configure a query on the left and run it to see results.
            </p>
          ) : (
            <pre className="max-h-[480px] overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/50 p-3 text-sm leading-relaxed font-sans">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
