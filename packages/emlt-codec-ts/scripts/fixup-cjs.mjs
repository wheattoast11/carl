// Post-build fixup for CJS output: emit a package.json marker so Node resolves
// the dist/cjs tree as CommonJS even though the package root declares ESM-first
// exports. This is the "dual package" pattern for Node >=18.
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const cjsDir = resolve(new URL(".", import.meta.url).pathname, "..", "dist", "cjs");
mkdirSync(cjsDir, { recursive: true });
writeFileSync(
  resolve(cjsDir, "package.json"),
  JSON.stringify({ type: "commonjs" }, null, 2) + "\n",
  "utf8",
);

const esmDir = resolve(new URL(".", import.meta.url).pathname, "..", "dist", "esm");
mkdirSync(esmDir, { recursive: true });
writeFileSync(
  resolve(esmDir, "package.json"),
  JSON.stringify({ type: "module" }, null, 2) + "\n",
  "utf8",
);
