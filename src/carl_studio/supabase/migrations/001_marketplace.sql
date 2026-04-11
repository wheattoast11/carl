-- Marketplace schema for carl.camp
-- Models, adapters, recipes, kits — shared training artifacts
create schema if not exists marketplace;

-- Models: trained model outputs
create table marketplace.models (
    id          uuid primary key default gen_random_uuid(),
    owner_id    uuid not null,
    hub_id      text not null unique,
    name        text not null,
    base_model  text not null,
    source_type text not null default 'repo',
    capability_dims text[] not null default '{}',
    description text not null default '',
    downloads   int not null default 0,
    stars       int not null default 0,
    public      boolean not null default false,
    metadata    jsonb not null default '{}',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Adapters: LoRA weights, mix-and-match
create table marketplace.adapters (
    id               uuid primary key default gen_random_uuid(),
    owner_id         uuid not null,
    hub_id           text not null unique,
    name             text not null,
    compatible_bases text[] not null default '{}',
    capability_dims  text[] not null default '{}',
    description      text not null default '',
    rank             int not null default 64,
    downloads        int not null default 0,
    stars            int not null default 0,
    public           boolean not null default false,
    metadata         jsonb not null default '{}',
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

-- Recipes: shared pipeline definitions
create table marketplace.recipes (
    id            uuid primary key default gen_random_uuid(),
    owner_id      uuid not null,
    name          text not null,
    slug          text not null unique,
    spec          jsonb not null,
    source_types  text[] not null default '{}',
    courses_count int not null default 0,
    description   text not null default '',
    stars         int not null default 0,
    public        boolean not null default false,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

-- Kits: shared reward configurations
create table marketplace.kits (
    id          uuid primary key default gen_random_uuid(),
    owner_id    uuid not null,
    name        text not null,
    slug        text not null unique,
    ingredients jsonb not null,
    molds       text[] not null default '{}',
    gates       jsonb not null default '[]',
    description text not null default '',
    stars       int not null default 0,
    public      boolean not null default false,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- RLS
alter table marketplace.models enable row level security;
alter table marketplace.adapters enable row level security;
alter table marketplace.recipes enable row level security;
alter table marketplace.kits enable row level security;

create policy "public_read_models" on marketplace.models for select using (public = true);
create policy "owner_read_models" on marketplace.models for select using (auth.uid() = owner_id);
create policy "owner_insert_models" on marketplace.models for insert with check (auth.uid() = owner_id);
create policy "owner_update_models" on marketplace.models for update using (auth.uid() = owner_id);

create policy "public_read_adapters" on marketplace.adapters for select using (public = true);
create policy "owner_read_adapters" on marketplace.adapters for select using (auth.uid() = owner_id);
create policy "owner_insert_adapters" on marketplace.adapters for insert with check (auth.uid() = owner_id);
create policy "owner_update_adapters" on marketplace.adapters for update using (auth.uid() = owner_id);

create policy "public_read_recipes" on marketplace.recipes for select using (public = true);
create policy "owner_read_recipes" on marketplace.recipes for select using (auth.uid() = owner_id);
create policy "owner_insert_recipes" on marketplace.recipes for insert with check (auth.uid() = owner_id);
create policy "owner_update_recipes" on marketplace.recipes for update using (auth.uid() = owner_id);

create policy "public_read_kits" on marketplace.kits for select using (public = true);
create policy "owner_read_kits" on marketplace.kits for select using (auth.uid() = owner_id);
create policy "owner_insert_kits" on marketplace.kits for insert with check (auth.uid() = owner_id);
create policy "owner_update_kits" on marketplace.kits for update using (auth.uid() = owner_id);

-- updated_at trigger
create or replace function marketplace.set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger models_updated before update on marketplace.models
    for each row execute function marketplace.set_updated_at();
create trigger adapters_updated before update on marketplace.adapters
    for each row execute function marketplace.set_updated_at();
create trigger recipes_updated before update on marketplace.recipes
    for each row execute function marketplace.set_updated_at();
create trigger kits_updated before update on marketplace.kits
    for each row execute function marketplace.set_updated_at();
