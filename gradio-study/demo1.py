#!/usr/bin/python3
# -*- coding: utf-8 -*-            
# @Author :le
# @Time : 2025/3/6 17:33

import gradio as gr
import numpy as np


def greet1(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def greet2(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2

demo = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    examples=[
        [45, "add", 3],
        [3.14, "divide", 2],
        [144, "multiply", 2.5],
        [0, "subtract", 1.2],
    ],
    title="Toy Calculator",
    description="Here's a sample toy calculator.",
)
# demo = gr.Interface(
    #     fn=greet1,
    #     inputs=[
    #         "text",
    #         gr.Slider(value=2, minimum=1, maximum=10, step=1)],
    #     outputs=[gr.Textbox(label="greeting", lines=3)]
    # )

# demo = gr.Interface(
#         fn=greet2,
#         inputs=[gr.Textbox(label="姓名"), gr.Checkbox(label="是否为早上"), gr.Slider(0, 100, label="温度")],
#         outputs=[gr.Textbox(label="欢迎"), "number"]
# )

# demo = gr.Interface(sepia, gr.Image(), "image")

if __name__ == "__main__":


    demo.launch()

