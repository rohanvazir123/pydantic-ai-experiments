
---
### <mark>FastAPI - POST/GET example<mark>
---
```
from typing import Generator

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
```

### <mark>1. Database Setup & Model<mark>
```
sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)

class Item(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    price: float
```

### <mark>2. Dependency Injection for Sessions<mark>
```
def get_session():
    with Session(engine) as session:
        yield session
```

### <mark>3. FastAPI App & Endpoints<mark>
```
app = FastAPI()

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.post("/items/", response_model=Item)
def create_item(item: Item, session: Session = Depends(get_session)):
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@app.get("/items/", response_model=list[Item])
def read_items(session: Session = Depends(get_session)):
    return session.exec(select(Item)).all()
```

Key Takeaways
* Zero Duplication: 
* Define database models and API schemas in one place, reducing code maintenance.
* IDE Support: Provides full type-hinting and autocomplete for database queries and Pydantic models.
* FastAPI Integration: Works natively with FastAPI for dependency injection and OpenAPI generation.

---
### <mark>Node.js express  - POST/GET example<mark>
---

* npm init -y
* npm install express dotenv
* npm install typescript @types/node @types/express ts-node-dev prisma @prisma/client --save-dev
* npx prisma init --datasource-provider sqlite

### <mark>1. Database Schema (prisma/schema.prisma)<mark>

* Define your database model. 
* Prisma automatically generates TypeScript types from this file.

```
datasource db {
  provider = "sqlite"
  url      = "file:./dev.db"
}

generator client {
  provider = "prisma-client-js"
}

model Item {
  id          Int     @id @default(autoincrement())
  name        String
  description String?
  price       Float
}
```

Run npx prisma db push in your terminal to create the SQLite file and generate the client library.

### <mark>2. Express Server (server.ts)<mark>
* This file instantiates the client, handles data parsing, and serves the POST and GET endpoints.

```
import express, { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const app = express();

// Middleware to parse incoming JSON payloads
app.use(express.json());

// CREATE Endpoint (POST)
app.post('/items', async (req: Request, res: Response): Promise<void> => {
  try {
    const { name, description, price } = req.body;

    // Simple validation block
    if (!name || typeof price !== 'number') {
      res.status(400).json({ error: "Missing required fields or invalid price" });
      return;
    }

    const newItem = await prisma.item.create({
      data: { name, description, price },
    });

    res.status(201).json(newItem);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// READ ALL Endpoint (GET)
app.get('/items', async (req: Request, res: Response): Promise<void> => {
  try {
    const items = await prisma.item.findMany();
    res.status(200).json(items);
  } catch (error) {
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```
