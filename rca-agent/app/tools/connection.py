import httpx, asyncio

async def test():
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://192.168.101.147:31998/query_log",
            headers={"Content-Type": "application/json"},
            json={"query_log": "test", "top_k": 3, "source": "", "alert_type": "error"}
        )
        print(r.status_code, r.text)

asyncio.run(test())