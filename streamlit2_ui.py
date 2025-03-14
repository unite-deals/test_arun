import io
import zipfile
from pathlib import Path
import streamlit as st
from PIL import Image
import uuid
from PyPDF2 import PdfReader
import cv2
import numpy as np
import fitz
import matplotlib.pyplot as plt
import shutil

MAX_FILES = 5
ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg"]

def setup_page():
    """Sets up the Streamlit page configuration and loads custom CSS."""
    st.set_page_config(page_title="Arraytree Slide Detection", page_icon="*")
    load_css()
    hide_streamlit_style()

def load_css():
    """Injects custom CSS with a glassmorphism theme for a modern UI."""
    css = """
    <style>
    /* Main container background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg,rgb(122, 168, 238) 0%, #c3cfe2 100%) !important;
    }
    /* Global Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Glassmorphism container style */
    .glass {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(22, 15, 146, 0.3) !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    /* Header Styles */
    .header {
        display: flex !important;
        align-items: center !important;
        padding: 10px 20px !important;
        background: rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        margin-bottom: 20px !important;
    }
    .header img {
        height: 50px !important;
        margin-right: 15px !important;
        border-radius: 8px !important;
    }
    .header h1 {
        margin: 0 !important;
        font-size: 24px !important;
        color: #fff !important;
    }
    /* Footer Styles */
    .footer {
        text-align: center !important;
        font-size: 14px !important;
        padding: 10px 0 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.3) !important;
        margin-top: 30px !important;
    }
    /* Sidebar Styles */
    [data-testid="stSidebar"] > div {
        background: rgba(228, 255, 173, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def hide_streamlit_style():
    """Hides default Streamlit styling."""
    st.markdown(
        "<style>footer {visibility: hidden;} #MainMenu {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )

def initialize_session():
    """Initializes a unique session ID."""
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = str(uuid.uuid4())

def display_header():
    """Displays a header with a logo and demo app title using glassmorphism style."""
    header_html = """
    <div class="header glass">
        <img src="https://i.imgur.com/4UjxpFN.png" alt="Logo">
        <h1> ArrayTree Slide Detection Demo App</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    st.warning("⚠️ Enter your output file initial seq number below:")
    user_input = st.text_input("Seq start number:")
    if user_input:
        st.success(f"Thank you!Check output folder once done")
    return user_input

def display_ui():
    """Displays the user interface for file upload and returns uploaded files."""
    st.sidebar.markdown("<div class='glass'><h3>Image Slide Generator</h3></div>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "file_uploader"),
    )
    return uploaded_files

def display_footer():
    """Displays a custom footer with neon deep green and bold contact information."""
    footer_html = """
    <div class="footer">
        <p style="color: #39FF14; font-weight: bold;">Arraytree Demo App | Contact: arraytree@gmail.com</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def process_and_display_images(uploaded_files, user_input):
    """Processes the uploaded files and displays the original and result images."""
    if not uploaded_files:
        st.warning("Please upload a file.")
        return

    if not st.sidebar.button("Slide Detection by AI"):
        return

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Maximum {MAX_FILES} files will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES]
    results = []

    with st.spinner("Detecting Slides..."):
        for uploaded_file in uploaded_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc[0]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # Reference to the image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                result_image = slide_detection(original_image, user_input)
                results.append((original_image, result_image, uploaded_file.name))
                zip_path = slide_detection_cutout(original_image, user_input)
                break  # Use only the first image per page

    for original, result, name in results:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original")
        with col2:
            st.image(result, caption="Result")
    with open(zip_path, "rb") as zip_file:
        st.download_button(
            label="Download Processed Images",
            data=zip_file,
            file_name="output_images.zip",
            mime="application/zip"
        )
    if len(results) > 1:
        download_zip(results)
    else:
        download_result(results[0])

def slide_detection(original_image, user_input):
    """Removes the background from an image."""
    original_image_cv = np.array(original_image)
    gray_image = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2GRAY)
    retval, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    _, im_with_separated_blobs, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    ht, wd = binary_image.shape
    #print(binary_image)
    flag = 0
    if binary_image[ht-100][100] == 0:
        inverted_image = cv2.bitwise_not(original_image_cv)
        inverted_gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((11, 11), np.uint8) 
        dil_image = cv2.dilate(binary_image, kernel, iterations=5)
        flag = 1
    else:
        inverted_image = original_image_cv
        inverted_gray_image = gray_image
        kernel = np.ones((11, 11), np.uint8) 
        retval, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        binary_image1 = 255 - binary_image
        dil_image = cv2.dilate(binary_image1, kernel, iterations=5)

    cropped = dil_image[int(ht/2-200):int(ht/2+200),:]  # Example cropping (adjust coordinates)
    _, _, stats_dil, _ = cv2.connectedComponentsWithStats(cropped)
    sorted_indices = np.argsort(stats_dil[:, 4])[::-1] 
    st_col1 = stats_dil[sorted_indices[1],0]
    end_col1 = stats_dil[sorted_indices[1],0] + stats_dil[sorted_indices[1],2]
    st_col2 =  None
    for i in sorted_indices:
        if i < 2:
            continue
        elif stats_dil[sorted_indices[i],2] > 100 and (stats_dil[sorted_indices[i],0] > end_col1 or stats_dil[sorted_indices[i],0]+stats_dil[sorted_indices[i],2] < st_col1):
            st_col2 = stats_dil[sorted_indices[i],0] 
            end_col2 = stats_dil[sorted_indices[i],0]  + stats_dil[sorted_indices[i],2] 
    centrss1 = []
    centrss2 = []
    min_size = 20
    max_size = 1000
    valid_labels = np.where((stats[:, cv2.CC_STAT_AREA] >= min_size) & (stats[:, cv2.CC_STAT_AREA] <= max_size))[0]

    blurred = cv2.GaussianBlur(inverted_gray_image, (13,13), 11)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=40,
        param2=30,
        minRadius=20,
        maxRadius=50
    )
   
    output_image = original_image_cv.copy()
    original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR) 
    #print(circles)
    h1, w1, r1 = original_image_cv.shape
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            mask = np.zeros(output_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255,), thickness=-1)
            # Compute mean pixel color inside the circle
            mean_color = cv2.mean(inverted_gray_image, mask=mask)  # Returns (B, G, R, Alpha)
            if mean_color[0] > 200:
                #cv2.circle(original_image_cv, (x, y), r, (255,), thickness=-1)
                if x < w1/2:
                    centrss1.append((x, y))
                else:
                    centrss2.append((x, y))
        circle_centers, approx_most_common_diff = slide_extract(centrss1, original_image_cv)
        xs = ys = 0
        ind = int(user_input)
        p11 = 0
        p11_bkp = 0
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for centr in circle_centers:        
                if xs == 0 and ys == 0:
                    xs, ys = centr
                    continue
                else:
                    xc, yc = centr
                    h = int((yc - ys) / 2)
                    if h>30:
                        if h <= int(approx_most_common_diff) + 6:             
                            p11 = yc - h
                            croppe = original_image_cv[int(p11_bkp):int(p11),st_col1:end_col1]
                            p11_bkp = p11
                            #cv2.imwrite(f"fcomp_{ind}.jpg", croppe)
                            buffer = io.BytesIO()
                            Image.fromarray(croppe).save(buffer, format="PNG")
                            zip_file.writestr(f"fslide_{ind}.png", buffer.getvalue())
                            ind = ind + 1
                            cv2.line(original_image_cv, (0, int(p11)), (int(w1/3), int(p11)), (255, 0, 0), 8)
                            xs, ys = xc, yc
                        else:  
                            p11 = ys + int(approx_most_common_diff)
                            cv2.line(original_image_cv, (0, int(p11)), (int(w1/3), int(p11)), (255, 0, 0), 8)
                            p12 = yc - int(approx_most_common_diff)
                            croppe = original_image_cv[int(p11):int(p12),st_col1:end_col1]
                            #cv2.imwrite(f"fcomp_{ind}.jpg", croppe)
                            buffer = io.BytesIO()
                            Image.fromarray(croppe).save(buffer, format="PNG")
                            zip_file.writestr(f"fslide_{ind}.png", buffer.getvalue())
                            ind = ind + 1
                            cv2.line(original_image_cv, (0, int(p12)), (int(w1/3), int(p12)), (0, 0, 255), 8)
                            xs, ys = xc, yc

            circle_centers, approx_most_common_diff = slide_extract(centrss2, original_image_cv)
            xs = ys = 0        
            p11 = 0
            p11_bkp = 0
            for centr in circle_centers:        
                if xs == 0 and ys == 0:
                    xs, ys = centr
                    continue
                else:
                    xc, yc = centr
                    h = int((yc - ys) / 2)
                    if h>30:
                        if h <= int(approx_most_common_diff) + 6:             
                            p11 = yc - h
                            pt2 = int(wd - 5)
                            cv2.line(original_image_cv, (int(w1/2), int(p11)), (pt2, int(p11)), (0, 255, 0), 8)
                            xs, ys = xc, yc
                            if st_col2 is not None:
                                croppe = original_image_cv[int(p11_bkp):int(p11), st_col2:end_col2]
                                p11_bkp = p11
                                #cv2.imwrite(f"bcomp_{ind}.jpg", croppe)
                                buffer = io.BytesIO()
                                Image.fromarray(croppe).save(buffer, format="PNG")
                                zip_file.writestr(f"fslide_{ind}.png", buffer.getvalue())
                                ind = ind + 1
                        else:  
                            p11 = ys + int(approx_most_common_diff)
                            pt2 = int(wd - 5)
                            cv2.line(original_image_cv, (int(w1/2), int(p11)), (pt2, int(p11)), (0, 255, 0), 8)
                            p12 = yc - int(approx_most_common_diff)
                            cv2.line(original_image_cv, (int(w1/2), int(p12)), (pt2, int(p12)), (0, 0, 255), 8)
                            xs, ys = xc, yc
                            if st_col2 is not None:
                                croppe = original_image_cv[ int(p11):int(p12),st_col2:end_col2]
                                #cv2.imwrite(f"bcomp_{ind}.jpg", croppe)
                                buffer = io.BytesIO()
                                Image.fromarray(croppe).save(buffer, format="PNG")
                                zip_file.writestr(f"fslide_{ind}.png", buffer.getvalue())
                                ind = ind + 1
    zip_buffer.seek(0)
    st.download_button(
        label="Download All Slides 🔥",
        data=zip_buffer,
        file_name="slides.zip",
        mime="application/zip"
    )
    output_image = Image.fromarray(original_image_cv)
    return output_image

def slide_extract(centrss, original_image_cv):
    circle_centers = sorted(centrss, key=lambda x: x[1])
    avg_height = [circle_centers[i][1] - circle_centers[i-1][1] for i in range(1, len(circle_centers))]
    num_bins = 2
    hist, bin_edges = np.histogram(avg_height, bins=num_bins)
    max_bin_idx = np.argmax(hist)
    approx_most_common_diff = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    return circle_centers, approx_most_common_diff

def img_to_bytes(img):
    """Converts an Image object to bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
def process_and_display_images(uploaded_files, user_input):
    """Processes the uploaded files and displays the original and result images."""
    if not uploaded_files:
        st.warning("Please upload a file.")
        return

    if not st.sidebar.button("Slide Detection by AI"):
        return

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Maximum {MAX_FILES} files will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES]
    results = []

    with st.spinner("Detecting Slides..."):
        for uploaded_file in uploaded_files:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = doc[0]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]  # Reference to the image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                result_image = slide_detection(original_image, user_input)
                results.append((original_image, result_image, uploaded_file.name))
                break  # Use only the first image per page

    for original, result, name in results:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original")
        with col2:
            st.image(result, caption="Result")

    if len(results) > 1:
        download_zip(results)
    else:
        download_result(results[0])

def slide_detection_cutout(original_image, user_input):
    """Removes the background from an image."""
    original_image_cv = np.array(original_image)
    gray_image = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2GRAY)
    retval, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    _, im_with_separated_blobs, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    ht, wd = binary_image.shape
    #print(binary_image)
    flag = 0
    if binary_image[ht-100][100] == 0:
        inverted_image = cv2.bitwise_not(original_image_cv)
        inverted_gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((11, 11), np.uint8) 
        dil_image = cv2.dilate(binary_image, kernel, iterations=5)
        flag = 1
    else:
        inverted_image = original_image_cv
        inverted_gray_image = gray_image
        kernel = np.ones((11, 11), np.uint8) 
        retval, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        binary_image1 = 255 - binary_image
        dil_image = cv2.dilate(binary_image1, kernel, iterations=5)

    cropped = dil_image[int(ht/2-200):int(ht/2+200),:]  # Example cropping (adjust coordinates)
    _, _, stats_dil, _ = cv2.connectedComponentsWithStats(cropped)
    sorted_indices = np.argsort(stats_dil[:, 4])[::-1] 
    st_col1 = stats_dil[sorted_indices[1],0]
    end_col1 = stats_dil[sorted_indices[1],0] + stats_dil[sorted_indices[1],2]
    st_col2 =  None
    for i in sorted_indices:
        if i < 2:
            continue
        elif stats_dil[sorted_indices[i],2] > 100 and (stats_dil[sorted_indices[i],0] > end_col1 or stats_dil[sorted_indices[i],0]+stats_dil[sorted_indices[i],2] < st_col1):
            st_col2 = stats_dil[sorted_indices[i],0] 
            end_col2 = stats_dil[sorted_indices[i],0]  + stats_dil[sorted_indices[i],2] 
    centrss1 = []
    centrss2 = []
    min_size = 20
    max_size = 1000
    valid_labels = np.where((stats[:, cv2.CC_STAT_AREA] >= min_size) & (stats[:, cv2.CC_STAT_AREA] <= max_size))[0]

    blurred = cv2.GaussianBlur(inverted_gray_image, (13,13), 11)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=40,
        param2=30,
        minRadius=20,
        maxRadius=50
    )
   
    output_image = original_image_cv.copy()
    original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR) 
    #print(circles)
    h1, w1, r1 = original_image_cv.shape
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            mask = np.zeros(output_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255,), thickness=-1)
            # Compute mean pixel color inside the circle
            mean_color = cv2.mean(inverted_gray_image, mask=mask)  # Returns (B, G, R, Alpha)
            if mean_color[0] > 200:
                #cv2.circle(original_image_cv, (x, y), r, (255,), thickness=-1)
                if x < w1/2:
                    centrss1.append((x, y))
                else:
                    centrss2.append((x, y))
        circle_centers, approx_most_common_diff = slide_extract(centrss1, original_image_cv)
        xs = ys = 0
        ind = int(user_input)
        p11 = 0
        p11_bkp = 0
        for centr in circle_centers:        
            if xs == 0 and ys == 0:
                xs, ys = centr
                continue
            else:
                xc, yc = centr
                h = int((yc - ys) / 2)
                if h>30:
                    if h <= int(approx_most_common_diff) + 6:             
                        p11 = yc - h
                        croppe = original_image_cv[int(p11_bkp):int(p11),st_col1:end_col1]
                        p11_bkp = p11
                        cv2.imwrite(f"fcomp_{ind}.jpg", croppe)
                        ind = ind + 1
                        cv2.line(original_image_cv, (0, int(p11)), (int(w1/3), int(p11)), (255, 0, 0), 8)
                        xs, ys = xc, yc
                    else:  
                        p11 = ys + int(approx_most_common_diff)
                        cv2.line(original_image_cv, (0, int(p11)), (int(w1/3), int(p11)), (255, 0, 0), 8)
                        p12 = yc - int(approx_most_common_diff)
                        croppe = original_image_cv[int(p11):int(p12),st_col1:end_col1]
                        cv2.imwrite(f"fcomp_{ind}.jpg", croppe)
                        ind = ind + 1
                        cv2.line(original_image_cv, (0, int(p12)), (int(w1/3), int(p12)), (0, 0, 255), 8)
                        xs, ys = xc, yc

        circle_centers, approx_most_common_diff = slide_extract(centrss2, original_image_cv)
        xs = ys = 0        
        p11 = 0
        p11_bkp = 0
        for centr in circle_centers:        
            if xs == 0 and ys == 0:
                xs, ys = centr
                continue
            else:
                xc, yc = centr
                h = int((yc - ys) / 2)
                if h>30:
                    if h <= int(approx_most_common_diff) + 6:             
                        p11 = yc - h
                        pt2 = int(wd - 5)
                        cv2.line(original_image_cv, (int(w1/2), int(p11)), (pt2, int(p11)), (0, 255, 0), 8)
                        xs, ys = xc, yc
                        if st_col2 is not None:
                            croppe = original_image_cv[int(p11_bkp):int(p11), st_col2:end_col2]
                            p11_bkp = p11
                            cv2.imwrite(f"bcomp_{ind}.jpg", croppe)
                            ind = ind + 1
                    else:  
                        p11 = ys + int(approx_most_common_diff)
                        pt2 = int(wd - 5)
                        cv2.line(original_image_cv, (int(w1/2), int(p11)), (pt2, int(p11)), (0, 255, 0), 8)
                        p12 = yc - int(approx_most_common_diff)
                        cv2.line(original_image_cv, (int(w1/2), int(p12)), (pt2, int(p12)), (0, 0, 255), 8)
                        xs, ys = xc, yc
                        if st_col2 is not None:
                            croppe = original_image_cv[ int(p11):int(p12),st_col2:end_col2]
                            cv2.imwrite(f"bcomp_{ind}.jpg", croppe)
                            ind = ind + 1
    
    zip_filename = "output_images.zip"
    output_dir="results_output"
    shutil.make_archive("output_images", "zip", output_dir)
    return zip_filename

def download_result(image):
    """Allows the user to download the result image."""
    _, result, name = image
    st.download_button(
        label="Download Result",
        data=img_to_bytes(result),
        file_name=f"{Path(name).stem}_nobg.png",
        mime="image/png",
    )

def download_zip(images):
    """Allows the user to download results as a ZIP file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for _, image, name in images:
            image_bytes = img_to_bytes(image)
            zip_file.writestr(f"{Path(name).stem}_nobg.png", image_bytes)

    st.download_button(
        label="Download All as ZIP",
        data=zip_buffer.getvalue(),
        file_name="background_removed_images.zip",
        mime="application/zip",
    )

def main():
    setup_page()
    initialize_session()
    user_input = display_header()
    uploaded_files = display_ui()
    process_and_display_images(uploaded_files, user_input)
    display_footer()

if __name__ == "__main__":
    main()
