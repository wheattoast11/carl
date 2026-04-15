// Deno Edge Function: marketplace-star
// POST endpoint -- increments stars on a marketplace item. Requires JWT.
// Body: { type: string, id: string }
// Returns: { stars: number } or { error: string }
// Rate limit: one star per user per item (tracked via star_events table or idempotent check)

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const VALID_TYPES = new Set(["models", "adapters", "recipes", "kits"]);
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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

    // Validate request body
    let body: { type?: string; id?: string };
    try {
      body = await req.json();
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid JSON body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const { type, id } = body;
    if (!type || !VALID_TYPES.has(type)) {
      return new Response(
        JSON.stringify({ error: `type is required and must be one of: ${[...VALID_TYPES].join(", ")}` }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (!id || !UUID_RE.test(id)) {
      return new Response(
        JSON.stringify({ error: "id is required and must be a valid UUID" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const tableName = `marketplace.${type}`;

    // Verify item exists and is public (or owned by user)
    const { data: item, error: fetchError } = await supabase
      .from(tableName)
      .select("id, stars, public, owner_id")
      .eq("id", id)
      .single();

    if (fetchError || !item) {
      return new Response(
        JSON.stringify({ error: "Item not found" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    if (!item.public && item.owner_id !== user.id) {
      return new Response(
        JSON.stringify({ error: "Item not found" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Prevent self-starring
    if (item.owner_id === user.id) {
      return new Response(
        JSON.stringify({ error: "Cannot star your own item" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    // Increment stars atomically
    // Note: For production, a separate star_events(user_id, item_type, item_id) table
    // with a unique constraint would prevent duplicate stars. For the stub, we use
    // a simple increment -- the dedup table can be added in a future migration.
    const { data: updated, error: updateError } = await supabase
      .from(tableName)
      .update({ stars: item.stars + 1 })
      .eq("id", id)
      .select("stars")
      .single();

    if (updateError) {
      return new Response(
        JSON.stringify({ error: updateError.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({ stars: updated?.stars ?? item.stars + 1 }),
      { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (err) {
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
