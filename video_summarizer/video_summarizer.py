import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from google.generativeai import upload_file,get_file
import google.generativeai as genai
from phi.tools.duckduckgo import DuckDuckGo
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()
import os
API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# page configuration
st.set_page_config(
    page_title=('multimodal ai agent - video summarizer')

)
st.title("phidata video AI  summarizer agent")


# agent 
@st.cache_resource
def initialize_agent():
    return Agent(
        name='video AI summarizer',
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
        debug_mode=True,
    )

multimodal_agent=initialize_agent()

# file upload
video_file=st.file_uploader("upload a video file", type=['mp4'])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name
    # st.video(video_path, format='video/mp4', start_time=0)

user_query=st.text_area(
    'what insights are you seeking from the video'
)

if st.button("analyze video", key="analyze_video_button"):
    if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
    else:
        try:
            with st.spinner("Processing video and gathering insights..."):
                processed_video = upload_file(video_path)
                while processed_video.state.name == "PROCESSING":
                    time.sleep(1)
                    processed_video = get_file(processed_video.name)
                analysis_prompt=(
                    f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """   
                )     
                response=multimodal_agent.run(analysis_prompt, videos=[processed_video])
            # Display the result
            st.subheader("Analysis Result")
            st.markdown(response.content) 

        except Exception as error:
            st.error(f"An error occurred during analysis: {error}")
        finally:
            # Clean up temporary video file
            Path(video_path).unlink(missing_ok=True)  

else:
    st.info("Upload a video file to begin analysis.")  


# Customize text area height
# st.markdown(
#     """
#     <style>
#     .stTextArea textarea {
#         height: 100px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )          