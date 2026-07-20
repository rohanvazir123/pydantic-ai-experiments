import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";

import * as schema from "./schema";

// `npx drizzle-kit push` (run manually, see README) creates the table for
// this demo. Production apps instead run `drizzle-kit generate` once to
// produce versioned SQL files, then call `migrate(db, { migrationsFolder })`
// from "drizzle-orm/better-sqlite3/migrator" here at startup — unlike
// Alembic, Drizzle's migrator is designed to run in-process.
// See ../README.md#migrations-in-production.
const sqlite = new Database("dev.db");
export const db = drizzle(sqlite, { schema });
