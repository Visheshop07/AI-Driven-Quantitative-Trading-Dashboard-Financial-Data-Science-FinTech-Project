import random

def generate_demo_orders():
    orders = []

    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE"]

    for i in range(5):
        order = {
            "Symbol": random.choice(symbols),
            "Type": random.choice(["BUY", "SELL"]),
            "Price": round(random.uniform(100, 500), 2),
            "Quantity": random.randint(1, 10),
            "Status": random.choice(["EXECUTED", "PENDING"])
        }
        orders.append(order)

    return orders