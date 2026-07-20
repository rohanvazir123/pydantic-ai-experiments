import express, { Request, Response } from "express";

import { db } from "./db/client";
import { items } from "./db/schema";

const app = express();

app.use(express.json());

app.post("/items", async (req: Request, res: Response): Promise<void> => {
  try {
    const { name, description, price } = req.body;

    if (!name || typeof price !== "number") {
      res.status(400).json({ error: "Missing required fields or invalid price" });
      return;
    }

    const [newItem] = await db.insert(items).values({ name, description, price }).returning();

    res.status(201).json(newItem);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.get("/items", async (req: Request, res: Response): Promise<void> => {
  try {
    const allItems = await db.select().from(items);
    res.status(200).json(allItems);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
