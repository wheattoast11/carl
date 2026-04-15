// Deno Edge Function: marketplace-publish
// POST endpoint -- upserts a marketplace item. Requires JWT + paid tier.
// Body: { type: string, item: { name, hub_id|slug, description, ... } }
// Returns: the upserted item or { error: string }

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const VALID_TYPES = new Set(["models", "adapters", "recipes", "kits"]);

// Minimal required fields per type (beyond owner_id which is set server-side)
const REQUIRED_FIELDS: Record<string, string[]> = {
  models: ["hub_id", "name", "base_model"],
  adapters: ["hub_id", "name"],
  recipes: ["name", "slug", "spec"],
  kits: ["name", "slug", "ingredients"],
};

// Unique constraint column per type (used for upsert conflict target)
const UNIQUE_COL: Record<string, string> = {
  models: "hub_id",
  adapters: "hub_id",
  recipes: "slug",
  kits: "slug",
};

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

    // Check tier -- publishing requires paid subscription
    const { data: profile, error: profileError } = await supabase
      .from("user_profiles")
      .select("tier")
      .eq("id", user.id)
      .single();

    if (profileError || !profile) {
      return new Response(
        JSON.stringify({ error: "User profile not found" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (profile.tier !== "paid") {
      return new Response(
        JSON.stringify({ error: "Publishing requires a paid subscription. Visit carl.camp/pricing" }),
        { status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Validate request body
    let body: { type?: string; item?: Record<string, unknown> };
    try {
      body = await req.json();
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid JSON body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const { type, item } = body;
    if (!type || !VALID_TYPES.has(type)) {
      return new Response(
        JSON.stringify({ error: `type is required and must be one of: ${[...VALID_TYPES].join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (!item || typeof item !== "object") {
      return new Response(
        JSON.stringify({ error: "item object is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Validate required fields
    const required = REQUIRED_FIELDS[type];
    const missing = required.filter((f) => !item[f]);
    if (missing.length > 0) {
      return new Response(
        JSON.stringify({ error: `Missing required fields: ${missing.join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Set owner_id server-side (cannot be spoofed)
    const row = { ...item, owner_id: user.id };

    // Remove fields the client should not set
    delete row.id;
    delete row.stars;
    delete row.downloads;
    delete row.created_at;

    // Upsert on unique constraint
    const tableName = `marketplace.${type}`;
    const uniqueCol = UNIQUE_COL[type];
    const { data: upserted, error: upsertError } = await supabase
      .from(tableName)
      .upsert(row, { onConflict: uniqueCol })
      .select()
      .single();

    if (upsertError) {
      // Check for ownership conflict (trying to upsert someone else's item)
      if (upsertError.message.includes("violates row-level security")) {
        return new Response(
          JSON.stringify({ error: "Cannot update items you don't own" }),
          { status: 403, headers: { ...corsHeaders, "Content-Type": "application/json" } },
        );
      }
      return new Response(
        JSON.stringify({ error: upsertError.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    return new Response(JSON.stringify(upserted), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
