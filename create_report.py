from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Project Report: American Sign Language Detection', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.chapter_title('1. Objective')
pdf.chapter_body(
    "The objective of this project is to build a system that can detect a given "
    "American Sign Language (ASL) input image and output what the sign represents "
    "(such as letters of the alphabet)."
)

pdf.chapter_title('2. Dataset Information')
pdf.chapter_body(
    "The dataset contains 29 classes of signs, of which 26 are for the letters A-Z "
    "and 3 classes are for SPACE, DELETE, and NOTHING. The dataset provides separate "
    "training and testing images to train a robust classification model."
)

pdf.chapter_title('3. Methodology')
pdf.chapter_body(
    "1. Data Preprocessing: Images were loaded, resized to 64x64 pixels, and normalized "
    "to values between 0 and 1. Data augmentation techniques like rotation, zoom, "
    "and shifting were applied to prevent overfitting.\n"
    "2. Model Architecture: A Convolutional Neural Network (CNN) was implemented "
    "using TensorFlow/Keras. The model consists of multiple Conv2D, MaxPooling2D, "
    "BatchNormalization, and Dropout layers for feature extraction.\n"
    "3. Model Training: The model was compiled with the Adam optimizer and "
    "categorical crossentropy loss. Early stopping was used to end training when "
    "performance stopped improving on the validation set.\n"
    "4. Output: The model outputs probabilities across 29 classes."
)

pdf.chapter_title('4. Results and Conclusion')
pdf.chapter_body(
    "The CNN model converged successfully and demonstrates a strong capability "
    "to distinguish between different ASL hand gestures. The model can accurately "
    "predict the corresponding alphabet for unseen images. The trained model is "
    "saved for further inference tasks."
)

pdf.output('ASL_Detection_Report.pdf')
