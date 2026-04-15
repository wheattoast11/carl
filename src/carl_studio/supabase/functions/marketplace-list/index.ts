// Deno Edge Function: marketplace-list
// GET endpoint -- lists marketplace items by type
// Query params: type (models|adapters|recipes|kits), q (text search), limit, offset, public (bool)
// No auth required for public items. Auth required to see own private items.
// Returns: { items: [...], total: number }

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const VALID_TYPES = new Set(["models", "adapters", "recipes", "kits"]);

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

    // Parse query params
    const url = new URL(req.url);
    const type = url.searchParams.get("type") ?? "models";
    if (!VALID_TYPES.has(type)) {
      return new Response(
        JSON.stringify({ error: `Invalid type. Must be one of: ${[...VALID_TYPES].join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const rawLimit = parseInt(url.searchParams.get("limit") ?? "20", 10);
    const limit = Math.min(Math.max(1, isNaN(rawLimit) ? 20 : rawLimit), 100);
    const rawOffset = parseInt(url.searchParams.get("offset") ?? "0", 10);
    const offset = Math.max(0, isNaN(rawOffset) ? 0 : rawOffset);
    const searchQuery = url.searchParams.get("q");
    const publicOnly = url.searchParams.get("public") !== "false";

    // Optional auth -- allows showing user's own private items alongside public
    let userId: string | null = null;
    const authHeader = req.headers.get("Authorization");
    if (authHeader?.startsWith("Bearer ")) {
      const jwt = authHeader.slice(7);
      const { data: { user } } = await supabase.auth.getUser(jwt);
      userId = user?.id ?? null;
    }

    // Build query against marketplace schema
    // Using service role client to bypass RLS, then filtering manually
    // This allows combining public items + user's own private items
    const tableName = `marketplace.${type}`;

    let query = supabase
      .from(tableName)
      .select("*", { count: "exact" })
      .order("stars", { ascending: false })
      .order("created_at", { ascending: false })
      .range(offset, offset + limit - 1);

    if (publicOnly && !userId) {
      // No auth: only public items
      query = query.eq("public", true);
    } else if (publicOnly && userId) {
      // Authed: public items + own private items
      query = query.or(`public.eq.true,owner_id.eq.${userId}`);
    } else if (!publicOnly && userId) {
      // Explicit private: only user's own items
      query = query.eq("owner_id", userId);
    } else {
      // No auth, not public -- nothing to show
      return new Response(
        JSON.stringify({ items: [], total: 0 }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Text search on name + description
    if (searchQuery) {
      query = query.or(`name.ilike.%${searchQuery}%,description.ilike.%${searchQuery}%`);
    }

    const { data: items, error, count } = await query;

    if (error) {
      return new Response(
        JSON.stringify({ error: error.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({ items: items ?? [], total: count ?? 0 }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
