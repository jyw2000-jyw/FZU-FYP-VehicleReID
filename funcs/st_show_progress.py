import streamlit as st
import time

# Create an empty element to display the progress bar and text side by side.
progress_container = st.empty()

# Define a function to simulate a long-running process.
def long_running_process():
    for i in range(101):
        # Update the value of the progress bar with each iteration.
        progress_bar.progress(i)
        # Update the text next to the progress bar with the current progress percentage or "Finish!" if i == 100.
        if i == 100:
            progress_text.text("Progress: Finish!")
        else:
            progress_text.text(f"Progress: {i}%")
        time.sleep(0.1)

# Create a progress bar object and set the initial value to 0.
progress_bar = st.progress(0)
# Add text next to the progress bar.
progress_text = progress_container.text("Progress: 0%")

# Call the long_running_process function to start the process.
long_running_process()
