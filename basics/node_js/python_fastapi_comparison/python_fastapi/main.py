from fastapi import Depends, FastAPI
from sqlmodel import Field, Session, SQLModel, create_engine, select

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)


class Item(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str | None = None
    price: float


def get_session():
    with Session(engine) as session:
        yield session


app = FastAPI()


@app.on_event("startup")
def on_startup() -> None:
    # create_all() only adds missing tables, never alters existing ones —
    # fine for a demo, not a real migration tool. Production apps swap this
    # for Alembic (`alembic upgrade head`, run as a deploy step or via
    # alembic.command.upgrade() here). See ../README.md#migrations-in-production.
    SQLModel.metadata.create_all(engine)


@app.post("/items/", response_model=Item)
def create_item(item: Item, session: Session = Depends(get_session)) -> Item:
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@app.get("/items/", response_model=list[Item])
def read_items(session: Session = Depends(get_session)) -> list[Item]:
    return list(session.exec(select(Item)).all())
