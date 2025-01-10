import os
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def select_image():
    file_path = filedialog.askopenfilename(
        title="Selectează imaginea",
        filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.tiff;*.jpeg")]
    )
    return file_path


def show_plot_in_cv2(plot_func, title="Plot"):
    fig = plt.figure()
    plot_func()
    plt.tight_layout()
    temp_file = "temp_plot.png"
    fig.savefig(temp_file)
    plt.close(fig)
    img = cv2.imread(temp_file)
    cv2.imshow(title, img)
    os.remove(temp_file)


#####LABORATOARE###############
def laborator_1():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine NU e selectată.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Imaginea nu există sau formatul este invalid.")
        return
    cv2.imshow('Imagine Originală', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
def laborator_2():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    directory = r'C:/Users/alex_/Desktop/pi'
    img = cv2.imread(image_path)
    if img is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, BlackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('Original Image', img)
    cv2.imshow('Gray Image', gray)
    cv2.imshow('Black & White Image', BlackAndWhiteImage)
    cv2.imshow('HSV Image', hsvImage)

    def show_gray_histogram():
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
        cv2.normalize(histogram, histogram, 0, 300, cv2.NORM_MINMAX)
        for x in range(1, 256):
            cv2.line(hist_img,
                     (int((x-1)*2), 300 - int(histogram[x-1])),
                     (int(x*2), 300 - int(histogram[x])),
                     (255, 255, 255), thickness=1)
        cv2.imshow("Histogramă Grayscale", hist_img)

    def show_color_histogram():
        hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
            cv2.normalize(histogram, histogram, 0, 300, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(hist_img,
                         (int((x-1)*2), 300 - int(histogram[x-1])),
                         (int(x*2), 300 - int(histogram[x])),
                         (255 if col == 'b' else 0, 255 if col == 'g' else 0, 255 if col == 'r' else 0), thickness=1)
        cv2.imshow("Histograma Color", hist_img)

    show_gray_histogram()
    show_color_histogram()
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)
    cv2.imwrite('SavedFlower_Gray.jpg', gray)
    cv2.imwrite('SavedFlower_BW.jpg', BlackAndWhiteImage)
    cv2.imwrite('SavedFlower_HSV.jpg', hsvImage)
    print('Imagini salvate cu succes.')
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def laborator_3():
        image_path = select_image()
        if not image_path:
            print("Nicio imagine selectată.")
            return

        img = cv2.imread(image_path)
        if img is None:
            print("Imaginea nu există sau formatul este invalid.")
            return

        negative_img = 255 - img
        contrast_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
        gamma = 3
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(img, table)
        brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=60)

        cv2.imshow("Imaginea originala", img)
        cv2.imshow('Negativarea imaginii', negative_img)
        cv2.imshow('Modificarea contrastului', contrast_img)
        cv2.imshow('Corectia gamma', gamma_img)
        cv2.imshow('Modificarea luminozitatii', brightness_img)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()



############# Funcții apelate in -LABORATOR 4
def contur_extragere(imagine):
    gri = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gri, (3, 3))
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    return gradient


def gaussian_blur(image, kernel_size, sigma):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = np.outer(kernel, kernel)
    return cv2.filter2D(image, -1, gaussian_kernel)


def umplere_regiuni(imagine):
    gri = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)
    _, binar = cv2.threshold(gri, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9, 9), np.uint8)
    closing = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel)
    return closing


def bidimensional_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def show_images(original, filtered1, filtered2):
    cv2.imshow("Original", original)
    cv2.imshow("Gaussian Blur", filtered1)
    cv2.imshow("Bidimensional Filter", filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def laborator_4():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    mean_kernel = np.ones((3, 3), np.float32) / 9
    img_mean_blur = cv2.filter2D(img, -1, mean_kernel)
    img_gaussian_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
    img_laplacian = cv2.Laplacian(img, cv2.CV_64F)
    custom_kernel = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])
    img_high_pass = cv2.filter2D(img, -1, custom_kernel)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Compararea filtrelor aplicate")

    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title("Originală")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_mean_blur, cmap='gray')
    axs[0, 1].set_title("Medie aritmetică (3x3)")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(img_gaussian_blur, cmap='gray')
    axs[0, 2].set_title("Gaussian Blur")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(img_laplacian, cmap='gray')
    axs[1, 0].set_title("Filtru Laplace")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(img_high_pass, cmap='gray')
    axs[1, 1].set_title("Kernel personalizat")
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')

    cv2.imshow("Originală", img)
    cv2.imshow("Medie aritmetică (3x3)", img_mean_blur)
    cv2.imshow("Gaussian Blur", img_gaussian_blur)
    cv2.imshow("Filtru Laplace", np.uint8(img_laplacian))
    cv2.imshow("Kernel personalizat", img_high_pass)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()



def laborator_5():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    kernel_size = 5
    sigma = 1.5
    start_time = time.time()
    gaussian_filtered = gaussian_blur(image, kernel_size, sigma)
    gaussian_time = time.time() - start_time
    print(f"Timp procesare Gaussian Blur: {gaussian_time:.5f} secunde")

    bidimensional_kernel = np.array([[1, 1, 1],
                                      [1, -8, 1],
                                      [1, 1, 1]])

    # Bidimensional Filter și măsurare timpului
    start_time = time.time()
    bidimensional_filtered = bidimensional_filter(image, bidimensional_kernel)
    bidimensional_time = time.time() - start_time
    print(f"Timp procesare Bidimensional Filter: {bidimensional_time:.5f} secunde")

    show_images(image, gaussian_filtered, bidimensional_filtered)



####lab6

from collections import deque

def connected_components_bfs(img):
    height, width = img.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label = 0

    for i in range(height):
        for j in range(width):
            if img[i, j] == 0 and labels[i, j] == 0:
                label += 1
                q = deque()
                q.append((i, j))
                labels[i, j] = label

                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and img[nx, ny] == 0 and labels[nx, ny] == 0:
                            labels[nx, ny] = label
                            q.append((nx, ny))
    return labels

def connected_components_two_pass(img):
    height, width = img.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label = 0
    edges = {}

    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                neighbors = []
                if i > 0 and labels[i - 1, j] > 0:
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:
                    neighbors.append(labels[i, j - 1])

                if not neighbors:
                    label += 1
                    labels[i, j] = label
                    edges[label] = []
                else:
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for neighbor in neighbors:
                        if neighbor != min_label:
                            edges[min_label].append(neighbor)
                            edges[neighbor].append(min_label)

    new_labels = np.zeros(label + 1, dtype=np.int32)
    new_label = 0

    for i in range(1, label + 1):
        if new_labels[i] == 0:
            new_label += 1
            q = deque([i])
            new_labels[i] = new_label

            while q:
                x = q.popleft()
                for y in edges.get(x, []):
                    if new_labels[y] == 0:
                        new_labels[y] = new_label
                        q.append(y)

    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                labels[i, j] = new_labels[labels[i, j]]

    return labels

def laborator_6():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # BFS Connected Components
    labels_bfs = connected_components_bfs(binary_image)

    # Two-Pass Connected Components
    labels_two_pass = connected_components_two_pass(binary_image)

    def visualize_labels(labels):
        unique_labels = np.unique(labels)
        colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for label in unique_labels:
            if label == 0:
                continue
            mask = (labels == label)
            color = np.random.randint(0, 255, size=3)
            colored_labels[mask] = color
        return colored_labels

    colored_bfs = visualize_labels(labels_bfs)
    colored_two_pass = visualize_labels(labels_two_pass)

    cv2.imshow("Imagine Originală", binary_image)
    cv2.imshow("BFS Algorithm", colored_bfs)
    cv2.imshow("Two-Pass Algorithm", colored_two_pass)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()



def laborator_7(image_path, window_size=15, C=2, low_ratio=0.5, high_ratio=0.2, blur_kernel=(3, 3)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

    img_blur = cv2.GaussianBlur(img, blur_kernel, 0)

    grad_x = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    mag_8u = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_map_adaptive = cv2.adaptiveThreshold(
        mag_8u, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        window_size,
        C
    )

    mag_8u_float = mag_8u.astype(np.float32)
    max_val = mag_8u_float.max()
    high_thresh = max_val * high_ratio
    low_thresh = high_thresh * low_ratio

    result = np.zeros_like(mag_8u, dtype=np.uint8)
    strong_i, strong_j = np.where(mag_8u_float > high_thresh)
    result[strong_i, strong_j] = 255

    weak_i, weak_j = np.where((mag_8u_float <= high_thresh) & (mag_8u_float >= low_thresh))
    weak_set = set(zip(weak_i, weak_j))

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    to_visit = list(zip(strong_i, strong_j))
    while to_visit:
        x, y = to_visit.pop()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < mag_8u.shape[0] and 0 <= ny < mag_8u.shape[1]:
                if (nx, ny) in weak_set:
                    result[nx, ny] = 255
                    weak_set.remove((nx, ny))
                    to_visit.append((nx, ny))

    return edge_map_adaptive, result

def laborator_7_button():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    try:
        adaptive_edges, final_edges = laborator_7(
            image_path, window_size=15, C=2, low_ratio=0.5, high_ratio=0.2
        )

        cv2.imshow("Adaptive Edge Map", adaptive_edges)
        cv2.imshow("Final Edges (Hysteresis)", final_edges)
        cv2.waitKey(0)

        cv2.imwrite("edges_adaptive.jpg", adaptive_edges)
        cv2.imwrite("edges_final_hysteresis.jpg", final_edges)
    except Exception as e:
        print(f"Error: {e}")


########################## TEME LABORATOR ###########################################

def tema_1():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    def detect_histogram_peaks(img, window_size=5, threshold=0.0003):
        histogram, bin_edges = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)
        normalized_hist = histogram / histogram.sum()
        peaks = []
        for idx in range(window_size, 256 - window_size):
            local_mean = np.mean(normalized_hist[idx - window_size: idx + window_size + 1])
            if (normalized_hist[idx] > local_mean + threshold and
                normalized_hist[idx] >= np.max(normalized_hist[idx - window_size: idx + window_size + 1])):
                peaks.append(idx)
        peaks = [0] + peaks + [255]
        return peaks, normalized_hist

    def apply_quantization(img, peaks):
        quantized_img = img.copy()
        rows, cols = img.shape
        for row in range(rows):
            for col in range(cols):
                pixel_val = img[row, col]
                nearest_peak = min(peaks, key=lambda p: abs(int(p) - int(pixel_val)))
                quantized_img[row, col] = nearest_peak
        return quantized_img

    def apply_floyd_steinberg_dithering(img, peaks):
        img = img.astype(float)
        rows, cols = img.shape
        for row in range(rows):
            for col in range(cols):
                current_pixel = img[row, col]
                quantized_pixel = min(peaks, key=lambda p: abs(p - current_pixel))
                img[row, col] = quantized_pixel
                error = current_pixel - quantized_pixel
                if col + 1 < cols:
                    img[row, col + 1] += error * 7 / 16
                if row + 1 < rows:
                    if col > 0:
                        img[row + 1, col - 1] += error * 3 / 16
                    img[row + 1, col] += error * 5 / 16
                    if col + 1 < cols:
                        img[row + 1, col + 1] += error * 1 / 16
        return np.clip(img, 0, 255).astype(np.uint8)

    peaks, normalized_hist = detect_histogram_peaks(img)
    print("Histogram peaks (threshold values):", peaks)

    quantized_img = apply_quantization(img, peaks)
    dithered_img = apply_floyd_steinberg_dithering(quantized_img, peaks)
    cv2.imshow("Original Image", img)
    cv2.imshow("Quantized Image", quantized_img)
    cv2.imshow("Dithered Image (Floyd-Steinberg)", dithered_img)

    def plot_histogram():
        plt.plot(normalized_hist, label="Histogramă Normalizată")
        plt.title("Histogramă Normalizată")
        plt.xlabel("Niveluri de gri")
        plt.ylabel("Frecvență")
        plt.legend()

    show_plot_in_cv2(plot_histogram, title="Histogramă")

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


def tema_2():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    def prag_binarizare_globala(image):
        valoare_prag = np.mean(image)
        _, imagine_binara = cv2.threshold(image, valoare_prag, 255, cv2.THRESH_BINARY)
        return imagine_binara, valoare_prag

    def egalizare_histograma(image):
        imagine_egalizata = cv2.equalizeHist(image)
        return imagine_egalizata

    imagine_binara, valoare_prag = prag_binarizare_globala(image)
    imagine_egalizata = egalizare_histograma(image)

    # Afișarea imaginilor
    cv2.imshow("Imagine originală", image)
    cv2.imshow(f"Imagine binarizată (Prag: {valoare_prag:.2f})", imagine_binara)
    cv2.imshow("Imagine egalizată", imagine_egalizata)

    # Funcții pentru histograme
    def plot_original_histogram():
        plt.title("Histogramă Originală")
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
        plt.xlabel("Niveluri de gri")
        plt.ylabel("Frecvență")

    def plot_equalized_histogram():
        plt.title("Histogramă Egalizată")
        plt.hist(imagine_egalizata.ravel(), bins=256, range=[0, 256], color='black')
        plt.xlabel("Niveluri de gri")
        plt.ylabel("Frecvență")

    show_plot_in_cv2(plot_original_histogram, title="Histogramă Originală")
    show_plot_in_cv2(plot_equalized_histogram, title="Histogramă Egalizată")

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
def tema_3():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    def compute_fourier_spectrum(img):
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)
        spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        return dft_shift, spectrum

    def apply_frequency_filter(img, filter_mask):
        dft_shift, _ = compute_fourier_spectrum(img)
        filtered_dft = dft_shift * filter_mask
        spectrum = 20 * np.log(np.abs(filtered_dft) + 1)
        inverse_dft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
        return np.abs(inverse_dft), spectrum

    def create_filter_mask(shape, filter_type, radius):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.float32)

        if filter_type == "low_pass":
            cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        elif filter_type == "high_pass":
            mask[:] = 1
            cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)

        return mask

    def create_gaussian_filter(shape, filter_type, sigma):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        x = np.arange(cols) - ccol
        y = np.arange(rows) - crow
        x, y = np.meshgrid(x, y)

        if filter_type == "low_pass":
            mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        elif filter_type == "high_pass":
            mask = 1 - np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return mask


    dft_shift, original_spectrum = compute_fourier_spectrum(img)

    radius = 30
    low_pass_mask = create_filter_mask(img.shape, "low_pass", radius)
    high_pass_mask = create_filter_mask(img.shape, "high_pass", radius)

    img_low_pass, spectrum_low_pass = apply_frequency_filter(img, low_pass_mask)
    img_high_pass, spectrum_high_pass = apply_frequency_filter(img, high_pass_mask)


    sigma = 30
    gaussian_low_pass = create_gaussian_filter(img.shape, "low_pass", sigma)
    gaussian_high_pass = create_gaussian_filter(img.shape, "high_pass", sigma)


    img_gaussian_low_pass, spectrum_gaussian_low_pass = apply_frequency_filter(img, gaussian_low_pass)
    img_gaussian_high_pass, spectrum_gaussian_high_pass = apply_frequency_filter(img, gaussian_high_pass)


    original_spectrum = cv2.normalize(original_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_low_pass = cv2.normalize(spectrum_low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_high_pass = cv2.normalize(spectrum_high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_gaussian_low_pass = cv2.normalize(spectrum_gaussian_low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_gaussian_high_pass = cv2.normalize(spectrum_gaussian_high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    cv2.imshow("Imagine Originală", img)
    cv2.imshow("Spectru Fourier Original", original_spectrum)
    cv2.imshow("Imagine Trece-Jos Circular", img_low_pass.astype(np.uint8))
    cv2.imshow("Spectru Trece-Jos Circular", spectrum_low_pass)
    cv2.imshow("Imagine Trece-Sus Circular", img_high_pass.astype(np.uint8))
    cv2.imshow("Spectru Trece-Sus Circular", spectrum_high_pass)
    cv2.imshow("Imagine Trece-Jos Gaussian", img_gaussian_low_pass.astype(np.uint8))
    cv2.imshow("Spectru Trece-Jos Gaussian", spectrum_gaussian_low_pass)
    cv2.imshow("Imagine Trece-Sus Gaussian", img_gaussian_high_pass.astype(np.uint8))
    cv2.imshow("Spectru Trece-Sus Gaussian", spectrum_gaussian_high_pass)

    key = cv2.waitKey(0)
    if key == 27:  # ESC pentru închidere
        cv2.destroyAllWindows()



def tema_4():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    imagine = cv2.imread(image_path)
    if imagine is None:
        print("Imaginea nu există sau formatul este invalid.")
        return

    contur = contur_extragere(imagine)
    umplere = umplere_regiuni(imagine)

    cv2.imshow("Imaginea Originală", imagine)
    cv2.imshow("Extragere Contur cu Morfologie", contur)
    cv2.imshow("Umplere Regiuni cu Morfologie", umplere)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()



root = tk.Tk()
root.title("PI")
root.geometry("800x600")
root.minsize(600, 400)

frame_content = tk.Frame(root, bg="lightblue", padx=20, pady=20)
frame_content.pack(fill="both", expand=True)

label_title = tk.Label(
    frame_content,
    text="Prelucrarea imaginilor",
    font=("Arial", 24, "bold"),
    bg="lightblue",
    fg="black"
)
label_title.pack(pady=20)


###tema5
def create_binary_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary

def contour_following(binary_image):
    contours = []
    visited = np.zeros_like(binary_image, dtype=bool)
    rows, cols = binary_image.shape

    for r in range(rows):
        for c in range(cols):
            if binary_image[r, c] == 0 and not visited[r, c]:
                contour = []
                start = (r, c)
                current = start
                prev_dir = 0

                while True:
                    contour.append(current)
                    visited[current] = True

                    found_next = False
                    for i in range(8):
                        next_dir = (prev_dir + i) % 8
                        dr, dc = direction_offset(next_dir)
                        nr, nc = current[0] + dr, current[1] + dc

                        if 0 <= nr < rows and 0 <= nc < cols and binary_image[nr, nc] == 0 and not visited[nr, nc]:
                            current = (nr, nc)
                            prev_dir = (next_dir + 4) % 8
                            found_next = True
                            break

                    if not found_next or current == start:
                        break

                contours.append(contour)

    return contours

def direction_offset(direction):
    offsets = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    return offsets[direction]

def chain_code(binary_image, contour):
    chain = []
    start = contour[0]
    current = start
    prev_dir = 0

    for _ in range(len(contour) - 1):
        for i in range(8):
            next_dir = (prev_dir + i) % 8
            dr, dc = direction_offset(next_dir)
            next_pixel = (current[0] + dr, current[1] + dc)

            if next_pixel in contour:
                chain.append(next_dir)
                current = next_pixel
                prev_dir = (next_dir + 4) % 8
                break

    return chain

def overlay_contours(binary_image, contours):
    overlay = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        for point in contour:
            overlay[point[0], point[1]] = (0, 255, 0)  # Green for contours
    return overlay
def tema_5():
    image_path = select_image()
    if not image_path:
        print("Nicio imagine selectată.")
        return

    binary_image = create_binary_image(image_path)
    contours = contour_following(binary_image)
    chain_codes = [chain_code(binary_image, contour) for contour in contours]

    overlay_image = overlay_contours(binary_image, contours)
    cv2.imshow("Contours Overlay", overlay_image)

    # Display chain codes in console
    for i, chain in enumerate(chain_codes):
        print(f"Chain Code for Contour {i + 1}: {chain}")

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


#Gradient fundal
def create_gradient(canvas, width, height, start_color, end_color):
    r1, g1, b1 = canvas.winfo_rgb(start_color)
    r2, g2, b2 = canvas.winfo_rgb(end_color)
    r_ratio = (r2 - r1) / height
    g_ratio = (g2 - g1) / height
    b_ratio = (b2 - b1) / height

    for i in range(height):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = f"#{nr:04x}{ng:04x}{nb:04x}"
        canvas.create_line(0, i, width, i, fill=color)



canvas = tk.Canvas(root, width=2000, height=1400)
canvas.pack(fill="both", expand=True)
create_gradient(canvas, 2000, 1400, "#87CEEB", "#4682B4")


frame_content = tk.Frame(canvas, bg="lightblue", padx=60, pady=60)
frame_content.place(relx=0.5, rely=0.5, anchor="center")


label_title = tk.Label(
    frame_content,
    text="Interfață Algoritmi Prelucrarea Imaginilor",
    font=("Arial", 18, "bold"),
    bg="lightblue",
    fg="black"
)
label_title.pack(pady=10)

frame_laboratoare = tk.LabelFrame(
    frame_content, text="Laboratoare", font=("Arial", 14),
    bg="lightblue", fg="black", padx=10, pady=10
)
frame_laboratoare.pack(fill="both", expand=True, side=tk.LEFT, padx=10, pady=10)

frame_teme = tk.LabelFrame(
    frame_content, text="Teme", font=("Arial", 14),
    bg="lightblue", fg="black", padx=30, pady=30
)
frame_teme.pack(fill="both", expand=True, side=tk.RIGHT, padx=20, pady=20)

######BUTOANE#######
btn_lab1 = tk.Button(
    frame_laboratoare,
    text="Laborator 1",
    command=laborator_1,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab1.pack(pady=5)

btn_lab2 = tk.Button(
    frame_laboratoare,
    text="Laborator 2",
    command=laborator_2,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab2.pack(pady=5)

btn_lab3 = tk.Button(
    frame_laboratoare,
    text="Laborator 3",
    command=laborator_3,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab3.pack(pady=5)

btn_lab4 = tk.Button(
    frame_laboratoare,
    text="Laborator 4",
    command=laborator_4,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab4.pack(pady=5)

btn_lab5 = tk.Button(
    frame_laboratoare,
    text="Laborator 5",
    command=laborator_5,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab5.pack(pady=5)

btn_lab6 = tk.Button(
    frame_laboratoare,
    text="Laborator 6",
    command=laborator_6,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab6.pack(pady=5)
btn_lab7 = tk.Button(
    frame_laboratoare,
    text="Laborator 7",
    command=laborator_7_button,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_lab7.pack(pady=5)
############### Butoane Teme Laborator
btn_tema1 = tk.Button(
    frame_teme,
    text="Tema 1",
    command=tema_1,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_tema1.pack(pady=5)

btn_tema2 = tk.Button(
    frame_teme,
    text="Tema 2",
    command=tema_2,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_tema2.pack(pady=5)

btn_tema3 = tk.Button(
    frame_teme,
    text="Tema 3",
    command=tema_3,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_tema3.pack(pady=5)

btn_tema4 = tk.Button(
    frame_teme,
    text="Tema 4",
    command=tema_4,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_tema4.pack(pady=5)

btn_tema5 = tk.Button(
    frame_teme,
    text="Tema 5",
    command=tema_5,
    bg="#1E90FF", fg="white", font=("Arial", 12),
    relief="flat", padx=10, pady=5
)
btn_tema5.pack(pady=5)
root.mainloop()
