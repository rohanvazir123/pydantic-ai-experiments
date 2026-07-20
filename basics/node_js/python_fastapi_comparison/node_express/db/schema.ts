import { integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const items = sqliteTable("Item", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  name: text("name").notNull(),
  description: text("description"),
  price: real("price").notNull(),
});
