

def get_weather(city):
    return city + '明天下小雨'


def none():
    return "不需要工具"


tools = {
    "get_weather": get_weather,
    "none": none,
}

result = tools["get_weather"](**args)
print(result)
