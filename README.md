# simple test

note, tested on Apple M2 Pro, using approx 4-5 GB of RAM, with Qwen 2.5VL 3B Instruct.

```
venv
python3 test.py
```

# Note to apple users

hugginface puts the models in ~/.cache/hugginface

in the snapshot preprocessor_config.json I had to change the 

`"image_processor_type": "Qwen2VLImageProcessor",`

changes to the sample code

```
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="mps",
    low_cpu_mem_usage=True)
```

note the `mps` and `eager` flags.


ref. https://github.com/QwenLM/Qwen2.5-VL/issues/777

ref. https://github.com/QwenLM/Qwen2.5-VL/issues/760


# Server

```
# make sure pip install uvicorn transformers

# don't use the .py extension
python -m uvicorn server:app --reload 
```

curl req

```
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@./assets/invoice.jpg" \
  -F "text=Extract from this invoice the invoiced items as a list, output to JSON"
```

```
time curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@./assets/invoice.jpg" \
  -F "text=details of the invoice, from and to, output to raw JSON"
```

result (see assets/invoice.jpg)

out of mem? -> image too large.

```
{
    "invoice": {
        "number": "12345",
        "date": "16 June 2025",
        "billed_to": {
            "name": "Imani Olowe",
            "address": "63 Ivy Road, Hawkville, GA, USA 31036",
            "phone": "+123-456-7890"
        },
        "items": [
            {
                "item": "Eggshell Camisole Top",
                "quantity": 1,
                "unit_price": 123,
                "total": 123
            },
            {
                "item": "Cuban Collar Shirt",
                "quantity": 2,
                "unit_price": 127,
                "total": 254
            },
            {
                "item": "Floral Cotton Dress",
                "quantity": 1,
                "unit_price": 123,
                "total": 123
            }
        ],
        "subtotal": 500,
        "tax": 0,
        "total": 500,
        "payment_information": {
            "bank": "Briard Bank",
            "account_name": "Samira Hadid",
            "account_number": "123-456-7890",
            "pay_by": "5 July 2025",
            "payee": "Samira Hadid",
            "payee_address": "123 Anywhere St., Any City, ST 12345"
        }
    }
}
```
