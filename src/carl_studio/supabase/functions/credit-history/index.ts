// Deno Edge Function: credit-history
// GET endpoint -- returns credit_transactions for the authenticated user
// Query params: limit (default 50, max 200), offset (default 0), type (filter by tx type)
// Returns: { transactions: [...], total: number, credits_remaining: number }

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const VALID_TYPES = new Set(["purchase", "included", "deduct", "refund", "adjustment"]);

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    if (!supabaseUrl || !serviceKey) {
      return new Response(
        JSON.stringify({ error: "Server misconfigured" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const supabase = createClient(supabaseUrl, serviceKey);

    // Verify JWT
    const authHeader = req.headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      return new Response(
        JSON.stringify({ error: "Missing auth" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }
    const jwt = authHeader.slice(7);
    const { data: { user }, error: authError } = await supabase.auth.getUser(jwt);
    if (authError || !user) {
      return new Response(
        JSON.stringify({ error: "Invalid token" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Parse query params
    const url = new URL(req.url);
    const rawLimit = parseInt(url.searchParams.get("limit") ?? "50", 10);
    const limit = Math.min(Math.max(1, isNaN(rawLimit) ? 50 : rawLimit), 200);
    const rawOffset = parseInt(url.searchParams.get("offset") ?? "0", 10);
    const offset = Math.max(0, isNaN(rawOffset) ? 0 : rawOffset);
    const typeFilter = url.searchParams.get("type");

    if (typeFilter && !VALID_TYPES.has(typeFilter)) {
      return new Response(
        JSON.stringify({ error: `Invalid type filter. Must be one of: ${[...VALID_TYPES].join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Fetch transactions
    let query = supabase
      .from("credit_transactions")
      .select("*", { count: "exact" })
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .range(offset, offset + limit - 1);

    if (typeFilter) {
      query = query.eq("type", typeFilter);
    }

    const { data: transactions, error: txError, count } = await query;

    if (txError) {
      return new Response(
        JSON.stringify({ error: txError.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Fetch current balance
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("credits_remaining, credits_total, credits_monthly_included")
      .eq("id", user.id)
      .single();

    if (profileError) {
      return new Response(
        JSON.stringify({ error: profileError.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        transactions: transactions ?? [],
        total: count ?? 0,
        credits_remaining: profile?.credits_remaining ?? 0,
        credits_total: profile?.credits_total ?? 0,
        credits_monthly_included: profile?.credits_monthly_included ?? 0,
      }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
