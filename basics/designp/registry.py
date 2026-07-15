from functools import wraps
from typing import Any, Callable

type Data = dict[str, Any]
type ExportFn = Callable[[Data], None]

export_funcs : dict[str, ExportFn] = {}

def register_export_func(format: str):
    def decorator(func: ExportFn):
        export_funcs[format] = func
        return func
    return decorator


@register_export_func("pdf")
def export_pdf(data: Data) -> None:
    print(f"Exporting data to PDF: {data}")

@register_export_func("csv")
def export_csv(data: Data) -> None:
    print(f"Exporting data to CSV: {data}")

print(export_funcs)

@register_export_func("json")
def export_json(data: Data) -> None:
    import json
    print("Exporting data to JSON:")
    print(json.dumps(data, indent=2))

print(export_funcs)

def export_data(data: Data, format: str):
    try:
        f = export_funcs[format]
        return f(data)
    except KeyError:
        print(f"format {format} is unsuppored")
    finally:
        pass



if __name__ == "__main__":

    export_data({'key1': 'val1'}, "pdf")
    export_data({'key2': 'val2'}, "csv")
    export_data({'key3': 'val3'}, "json")
    export_data({'key4': 'val4'}, "xml")
