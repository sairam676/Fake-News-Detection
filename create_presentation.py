import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="Fake_Image_Detection_Major_Project.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Clean Color Palette
    NAVY = RGBColor(24, 43, 73)         # #182B49 Dark Navy
    BLUE = RGBColor(37, 99, 235)        # #2563EB Vibrant Blue
    DARK = RGBColor(30, 41, 59)         # #1E293B Slate Dark Text
    GRAY = RGBColor(100, 116, 139)      # #64748B Subtle Gray
    LIGHT_BG = RGBColor(248, 250, 252)  # #F8FAFC Clean Light Canvas
    WHITE = RGBColor(255, 255, 255)
    CARD_BORDER = RGBColor(203, 213, 225) # #CBD5E1
    GREEN = RGBColor(16, 185, 129)      # #10B981 Emerald
    LIGHT_BLUE = RGBColor(238, 242, 255) # Light Indigo/Blue Card
    LIGHT_GREEN = RGBColor(236, 253, 245)
    RED = RGBColor(220, 38, 38)

    blank_layout = prs.slide_layouts[6]

    def set_bg(slide, color=LIGHT_BG):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="MAJOR PROJECT"):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(12)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY
        p_title.space_before = Pt(3)

    def add_card(slide, left, top, width, height, bg_color=WHITE, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
        return shape

    # =============================================================
    # SLIDE 1: TITLE SLIDE
    # =============================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, NAVY)

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.33), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "MAJOR PROJECT PRESENTATION"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 197, 253)

    p_main = tf1.add_paragraph()
    p_main.text = "AI-Generated Fake Image Detection &\nGenerator Attribution"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = WHITE
    p_main.space_before = Pt(12)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Comparing CNN Baseline (ResNet-18) with Multi-Domain Forensic Feature Models"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    p_sub.space_before = Pt(12)

    # Info card
    card_info = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.8), Inches(11.33), Inches(1.9))
    card_info.fill.solid()
    card_info.fill.fore_color.rgb = RGBColor(30, 58, 102)
    card_info.line.color.rgb = RGBColor(59, 130, 246)
    card_info.line.width = Pt(1.5)

    tf_info = card_info.text_frame
    tf_info.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_info.word_wrap = True

    p1 = tf_info.paragraphs[0]
    p1.text = "Project Team & Guide Details"
    p1.font.bold = True
    p1.font.size = Pt(16)
    p1.font.color.rgb = RGBColor(191, 219, 254)

    p2 = tf_info.add_paragraph()
    p2.text = "• Student Name(s): [Your Name / Team Members]      • Department: Computer Science & Engineering"
    p2.font.size = Pt(14)
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(6)

    p3 = tf_info.add_paragraph()
    p3.text = "• Guided By: [Supervisor / Professor Name]              • Academic Year: 2025 – 2026"
    p3.font.size = Pt(14)
    p3.font.color.rgb = WHITE
    p3.space_before = Pt(4)

    # =============================================================
    # SLIDE 2: PRESENTATION OUTLINE
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, LIGHT_BG)
    add_header(s2, "Presentation Outline", "ROADMAP")

    outline_items = [
        ("1", "Motivation & Introduction", "Why detecting AI fakes is crucial today"),
        ("2", "Literature Review & Gaps", "Limitations of current CNN models"),
        ("3", "Problem & Objectives", "Goals of our major project"),
        ("4", "Proposed Method & Flowchart", "4-branch feature extraction pipeline"),
        ("5", "Experimental Setup", "ResNet-18 baseline vs 5 feature models"),
        ("6", "Results & Key Findings", "Benchmark accuracy & attribution results"),
        ("7", "Conclusion & Future Scope", "Summary of work & next steps"),
        ("8", "References", "Key academic literature citations")
    ]

    for idx, (num, title, sub) in enumerate(outline_items):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 5.9)
        top = Inches(1.7 + row * 1.3)
        add_card(s2, left, top, Inches(5.7), Inches(1.15))

        tb = s2.shapes.add_textbox(left + Inches(0.2), top + Inches(0.12), Inches(5.3), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = f"{num}.  {title}"
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = NAVY if col == 0 else BLUE

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(13)
        p_s.font.color.rgb = GRAY
        p_s.space_before = Pt(2)

    # =============================================================
    # SLIDE 3: MOTIVATION BEHIND THE WORK
    # =============================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, LIGHT_BG)
    add_header(s3, "Motivation Behind the Work", "WHY IT MATTERS")

    mots = [
        ("Hyper-Realistic AI Images", "Diffusion & GAN models generate fake images that human eyes cannot spot.", BLUE),
        ("Spread of Misinformation", "Fake imagery threatens digital trust, journalism, identity, and security.", NAVY),
        ("Standard CNN Limitations", "CNNs (like ResNet-18) overfit to raw pixels and fail on unseen generators.", RED),
        ("Need for Attribution", "Crucial to identify not just IF an image is fake, but WHICH model created it.", GREEN)
    ]

    for idx, (title, desc, colr) in enumerate(mots):
        col = idx % 2
        row = idx // 2
        left = Inches(0.8 + col * 5.9)
        top = Inches(1.7 + row * 2.6)
        add_card(s3, left, top, Inches(5.7), Inches(2.3))

        tb = s3.shapes.add_textbox(left + Inches(0.3), top + Inches(0.2), Inches(5.1), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = f"✔  {title}"
        p.font.bold = True
        p.font.size = Pt(18)
        p.font.color.rgb = colr

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(15)
        p_d.font.color.rgb = DARK
        p_d.space_before = Pt(10)

    # =============================================================
    # SLIDE 4: INTRODUCTION
    # =============================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, LIGHT_BG)
    add_header(s4, "Introduction: Invisible Signatures in AI Images", "CORE CONCEPTS")

    intro_points = [
        ("1. Semantic Inconsistencies", "AI models often struggle with complex anatomy, physics, and symmetry (captured via DINOv2)."),
        ("2. Texture & Style Patterns", "Generative upsamplers leave repetitive color channel correlations (captured via Style Gram matrices)."),
        ("3. Frequency Grid Artifacts", "Latent decoders create periodic frequency anomalies (captured via 2D-FFT spectra)."),
        ("4. Camera Sensor Noise Absence", "Real cameras leave sensor noise (PRNU), whereas AI images lack natural sensor residuals.")
    ]

    for idx, (h, t) in enumerate(intro_points):
        top = Inches(1.7 + idx * 1.3)
        add_card(s4, Inches(0.8), top, Inches(11.7), Inches(1.15))

        tb = s4.shapes.add_textbox(Inches(1.0), top + Inches(0.12), Inches(11.3), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = BLUE

        p_t = tf.add_paragraph()
        p_t.text = t
        p_t.font.size = Pt(14)
        p_t.font.color.rgb = DARK
        p_t.space_before = Pt(2)

    # =============================================================
    # SLIDE 5: LITERATURE REVIEW & IDENTIFIED GAPS
    # =============================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, LIGHT_BG)
    add_header(s5, "Literature Review & Identified Gaps", "BACKGROUND & RESEARCH GAPS")

    # Left: Literature
    add_card(s5, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_l = s5.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Literature Review (Key Works):"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = NAVY

    lit = [
        ("Wang et al. (CVPR 2020):", "Showed CNNs trained on ProGAN can detect fakes, but fail on diffusion models."),
        ("Frank et al. (ICML 2020):", "Discovered frequency grid artifacts in generator upsampling layers using 2D-FFT."),
        ("DINOv2 (Oquab et al. 2023):", "Self-supervised ViT provides highly transferable semantic visual features.")
    ]
    for h, t in lit:
        p_item = tf_l.add_paragraph()
        p_item.space_before = Pt(10)
        r1 = p_item.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.size = Pt(14)
        r1.font.color.rgb = BLUE
        r2 = p_item.add_run()
        r2.text = t
        r2.font.size = Pt(14)
        r2.font.color.rgb = DARK

    # Right: Gaps
    add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_r = s5.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p_rg = tf_r.paragraphs[0]
    p_rg.text = "Identified Research Gaps:"
    p_rg.font.bold = True
    p_rg.font.size = Pt(18)
    p_rg.font.color.rgb = RED

    gaps = [
        ("Single-Domain Brittleness:", "Relying only on pixels or only on FFT fails when images are compressed or resized."),
        ("Generator Domain Shift:", "Detectors trained on GANs break down on modern Diffusion models (Midjourney, SD)."),
        ("Closed-Set Trap:", "Existing models cannot identify unseen or unknown new AI generators.")
    ]
    for h, t in gaps:
        p_gap = tf_r.add_paragraph()
        p_gap.space_before = Pt(10)
        r1 = p_gap.add_run()
        r1.text = f"✖ {h} "
        r1.font.bold = True
        r1.font.size = Pt(14)
        r1.font.color.rgb = RED
        r2 = p_gap.add_run()
        r2.text = t
        r2.font.size = Pt(14)
        r2.font.color.rgb = DARK

    # =============================================================
    # SLIDE 6: PROBLEM STATEMENT & OBJECTIVES
    # =============================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, LIGHT_BG)
    add_header(s6, "Problem Statement & Project Objectives", "GOALS & SCOPE")

    add_card(s6, Inches(0.8), Inches(1.7), Inches(11.7), Inches(1.4), bg_color=LIGHT_BLUE, border_color=BLUE)
    tb_p = s6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.2))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    p = tf_p.paragraphs[0]
    p.text = "Problem Statement:"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = BLUE
    p2 = tf_p.add_paragraph()
    p2.text = "Build a robust forensic system to accurately detect AI fake images and attribute them to their source generators by combining multi-domain features."
    p2.font.size = Pt(15)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.space_before = Pt(4)

    add_card(s6, Inches(0.8), Inches(3.3), Inches(11.7), Inches(3.6))
    tb_o = s6.shapes.add_textbox(Inches(1.0), Inches(3.45), Inches(11.3), Inches(3.3))
    tf_o = tb_o.text_frame
    tf_o.word_wrap = True
    p_o = tf_o.paragraphs[0]
    p_o.text = "Core Project Objectives:"
    p_o.font.bold = True
    p_o.font.size = Pt(17)
    p_o.font.color.rgb = NAVY

    objs = [
        ("1. CNN Baseline:", "Train ResNet-18 on raw RGB images as standard benchmark."),
        ("2. Multi-Domain Feature Extraction:", "Extract 4 features: DINOv2 (Semantic), Style Gram (Texture), 2D-FFT (Frequency), and Noise Residuals."),
        ("3. Ablation Study:", "Build and compare 5 feature-based models against ResNet-18."),
        ("4. Open-Set Attribution:", "Implement confidence thresholding to flag unknown new generators.")
    ]
    for h, t in objs:
        p_item = tf_o.add_paragraph()
        p_item.space_before = Pt(8)
        r1 = p_item.add_run()
        r1.text = f"✔ {h} "
        r1.font.bold = True
        r1.font.size = Pt(14.5)
        r1.font.color.rgb = BLUE
        r2 = p_item.add_run()
        r2.text = t
        r2.font.size = Pt(14.5)
        r2.font.color.rgb = DARK

    # =============================================================
    # SLIDE 7: SIMPLE & CLEAR FLOWCHART (PROPOSED METHOD)
    # =============================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, LIGHT_BG)
    add_header(s7, "Proposed Method: System Flowchart", "SIMPLE ARCHITECTURE PIPELINE")

    # Step 1: Input Box (Left)
    c_in = add_card(s7, Inches(0.8), Inches(3.1), Inches(2.0), Inches(2.2), bg_color=WHITE, border_color=BLUE)
    tb = s7.shapes.add_textbox(Inches(0.85), Inches(3.3), Inches(1.9), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "STEP 1\nInput Image"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = "RGB Image\n(224 × 224)"
    p2.font.size = Pt(13)
    p2.font.color.rgb = GRAY
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(6)

    # Arrow 1 -> 2
    a1 = s7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(2.9), Inches(3.9), Inches(0.5), Inches(0.4))
    a1.fill.solid(); a1.fill.fore_color.rgb = BLUE; a1.line.fill.background()

    # Step 2: 4 Feature Extractors (Middle-Left stacked)
    extractors = [
        ("DINOv2 ViT-S", "384-d", "Spatial / Semantic"),
        ("Style Gram", "64-d", "Texture Patterns"),
        ("2D-FFT", "64-d", "Frequency Grids"),
        ("Noise Stats", "64-d", "Sensor Residuals")
    ]
    for i, (name, dim, purpose) in enumerate(extractors):
        top_pos = Inches(1.7 + i * 1.3)
        add_card(s7, Inches(3.5), top_pos, Inches(2.8), Inches(1.15), bg_color=LIGHT_BLUE, border_color=BLUE)
        tb_e = s7.shapes.add_textbox(Inches(3.55), top_pos + Inches(0.1), Inches(2.7), Inches(0.95))
        tf_e = tb_e.text_frame
        tf_e.word_wrap = True
        pe = tf_e.paragraphs[0]
        pe.text = f"{name} ({dim})"
        pe.font.bold = True
        pe.font.size = Pt(14)
        pe.font.color.rgb = NAVY
        pe2 = tf_e.add_paragraph()
        pe2.text = purpose
        pe2.font.size = Pt(12)
        pe2.font.color.rgb = DARK

    # Arrow 2 -> 3
    a2 = s7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(6.4), Inches(3.9), Inches(0.5), Inches(0.4))
    a2.fill.solid(); a2.fill.fore_color.rgb = BLUE; a2.line.fill.background()

    # Step 3: Feature Concatenation (Middle-Right)
    add_card(s7, Inches(7.0), Inches(2.9), Inches(2.2), Inches(2.5), bg_color=WHITE, border_color=GREEN)
    tb_fc = s7.shapes.add_textbox(Inches(7.05), Inches(3.1), Inches(2.1), Inches(2.1))
    tf_fc = tb_fc.text_frame
    tf_fc.word_wrap = True
    pf = tf_fc.paragraphs[0]
    pf.text = "STEP 3\nFeature Fusion"
    pf.font.bold = True
    pf.font.size = Pt(16)
    pf.font.color.rgb = GREEN
    pf.alignment = PP_ALIGN.CENTER
    pf2 = tf_fc.add_paragraph()
    pf2.text = "Combined\n576-d Vector\n\n(Multi-Domain Signature)"
    pf2.font.size = Pt(13)
    pf2.font.color.rgb = DARK
    pf2.alignment = PP_ALIGN.CENTER
    pf2.space_before = Pt(6)

    # Arrow 3 -> 4
    a3 = s7.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(9.3), Inches(3.9), Inches(0.5), Inches(0.4))
    a3.fill.solid(); a3.fill.fore_color.rgb = GREEN; a3.line.fill.background()

    # Step 4: Classifier & Output (Right)
    add_card(s7, Inches(9.9), Inches(2.7), Inches(2.6), Inches(2.9), bg_color=NAVY, border_color=BLUE)
    tb_out = s7.shapes.add_textbox(Inches(9.95), Inches(2.85), Inches(2.5), Inches(2.6))
    tf_out = tb_out.text_frame
    tf_out.word_wrap = True
    po = tf_out.paragraphs[0]
    po.text = "STEP 4: Output"
    po.font.bold = True
    po.font.size = Pt(16)
    po.font.color.rgb = WHITE
    po.alignment = PP_ALIGN.CENTER

    po1 = tf_out.add_paragraph()
    po1.text = "1. Real vs. Fake\n(Binary Decision)"
    po1.font.bold = True
    po1.font.size = Pt(13)
    po1.font.color.rgb = RGBColor(147, 197, 253)
    po1.space_before = Pt(6)

    po2 = tf_out.add_paragraph()
    po2.text = "2. Generator Attribution\n(Midjourney, SD, BigGAN)"
    po2.font.bold = True
    po2.font.size = Pt(13)
    po2.font.color.rgb = RGBColor(167, 243, 208)
    po2.space_before = Pt(6)

    # =============================================================
    # SLIDE 8: EXPERIMENTAL SETUP & 6 BENCHMARK MODELS
    # =============================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8, LIGHT_BG)
    add_header(s8, "Experimental Setup: 6 Models Evaluated", "ABLATION DESIGN")

    # Clean Table
    rows, cols = 7, 4
    t_shape = s8.shapes.add_table(rows, cols, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    table = t_shape.table

    table.columns[0].width = Inches(1.6)
    table.columns[1].width = Inches(3.6)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(5.1)

    headers = ["Model", "Features / Architecture", "Dim", "Ablation Purpose"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE

    model_rows = [
        ("ResNet-18", "Standard CNN on Raw RGB", "512-d", "Classic deep learning baseline (Wang et al.)"),
        ("Baseline 2", "DINOv2 ViT-S CLS Token", "384-d", "Modern Vision Transformer baseline"),
        ("Model A", "Low-Bit Noise Residuals", "64-d", "Test if pixel noise alone is enough"),
        ("Model B", "DINOv2 + Style Gram", "448-d", "Test adding texture & color correlations"),
        ("Model C", "DINOv2 + Style + 2D-FFT", "512-d", "Test adding frequency grid features"),
        ("Proposed", "Full Multi-Branch Fused", "576-d", "Our complete 4-domain forensic model")
    ]

    for r_idx, row in enumerate(model_rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx == 5:
                cell.fill.fore_color.rgb = LIGHT_GREEN
            else:
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13)
            if r_idx == 5 or c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = RGBColor(22, 101, 52) if r_idx == 5 else NAVY
            else:
                p.font.color.rgb = DARK

    # =============================================================
    # SLIDE 9: EXPERIMENTAL RESULTS (CIFAKE BENCHMARK)
    # =============================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9, LIGHT_BG)
    add_header(s9, "Experimental Results: CIFAKE Benchmark", "REAL VS. SYNTHETIC EVALUATION")

    # Table on left (7.0 width)
    t_shape9 = s9.shapes.add_table(7, 4, Inches(0.8), Inches(1.7), Inches(6.8), Inches(5.2))
    t9 = t_shape9.table
    t9.columns[0].width = Inches(2.0)
    t9.columns[1].width = Inches(1.2)
    t9.columns[2].width = Inches(1.8)
    t9.columns[3].width = Inches(1.8)

    h9 = ["Model", "Dim", "Accuracy", "Macro F1"]
    for c_idx, h in enumerate(h9):
        cell = t9.cell(0, c_idx)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE

    res_data = [
        ("ResNet-18 (CNN)", "512", "87.50%", "0.8749"),
        ("Baseline (DINOv2)", "384", "92.75%", "0.9275"),
        ("Model A (Noise)", "64", "63.00%", "0.6297"),
        ("Model B (+Style)", "448", "93.00%", "0.9300"),
        ("Model C (+FFT)", "512", "92.00%", "0.9200"),
        ("Proposed (All)", "576", "94.00%", "0.9400")
    ]

    for r_idx, row in enumerate(res_data):
        for c_idx, val in enumerate(row):
            cell = t9.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if r_idx == 5:
                cell.fill.fore_color.rgb = LIGHT_GREEN
            else:
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13.5)
            if r_idx == 5:
                p.font.bold = True
                p.font.color.rgb = RGBColor(22, 101, 52)
            else:
                p.font.color.rgb = DARK

    # Card on right: Key Findings
    add_card(s9, Inches(7.9), Inches(1.7), Inches(4.6), Inches(5.2))
    tb_res = s9.shapes.add_textbox(Inches(8.1), Inches(1.9), Inches(4.2), Inches(4.8))
    tf_res = tb_res.text_frame
    tf_res.word_wrap = True

    p = tf_res.paragraphs[0]
    p.text = "Key Insights:"
    p.font.bold = True
    p.font.size = Pt(17)
    p.font.color.rgb = NAVY

    insights = [
        ("Proposed Model Wins:", "94.00% accuracy (+6.50% higher than ResNet-18 baseline)."),
        ("DINOv2 Superiority:", "Semantic ViT tokens outperform traditional CNN pixels by +5.25%."),
        ("Noise Alone Fails:", "Model A drops to 63.00%, proving pixel noise alone is vulnerable."),
        ("Fusion Benefit:", "Combining all 4 domains gives the highest detection consistency.")
    ]
    for h, t in insights:
        p_ins = tf_res.add_paragraph()
        p_ins.space_before = Pt(10)
        r1 = p_ins.add_run()
        r1.text = f"✔ {h} "
        r1.font.bold = True
        r1.font.size = Pt(13.5)
        r1.font.color.rgb = BLUE
        r2 = p_ins.add_run()
        r2.text = t
        r2.font.size = Pt(13)
        r2.font.color.rgb = DARK

    # =============================================================
    # SLIDE 10: MULTI-CLASS ATTRIBUTION & OPEN-SET DETECTION
    # =============================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10, LIGHT_BG)
    add_header(s10, "Generator Attribution & Open-Set Handling", "MULTI-GENERATOR BENCHMARK")

    # Left Card
    add_card(s10, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_att = s10.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_att = tb_att.text_frame
    tf_att.word_wrap = True

    p = tf_att.paragraphs[0]
    p.text = "1. Multi-Generator Attribution"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = NAVY

    p_classes = tf_att.add_paragraph()
    p_classes.text = "Tested across 5 image classes:\n• Real ImageNet, BigGAN, Midjourney v5,\n  Stable Diffusion v1.5, Wukong."
    p_classes.font.size = Pt(14)
    p_classes.font.color.rgb = DARK
    p_classes.space_before = Pt(8)

    p_r18 = tf_att.add_paragraph()
    p_r18.text = "• ResNet-18 Baseline: 81.25% Accuracy"
    p_r18.font.bold = True
    p_r18.font.size = Pt(14)
    p_r18.font.color.rgb = RED
    p_r18.space_before = Pt(10)

    p_prop = tf_att.add_paragraph()
    p_prop.text = "• Proposed Fused Model: 96.88% Accuracy"
    p_prop.font.bold = True
    p_prop.font.size = Pt(14.5)
    p_prop.font.color.rgb = RGBColor(22, 101, 52)
    p_prop.space_before = Pt(6)

    # Right Card
    add_card(s10, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_op = s10.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_op = tb_op.text_frame
    tf_op.word_wrap = True

    p_op = tf_op.paragraphs[0]
    p_op.text = "2. Open-Set 'Unknown' Detection"
    p_op.font.bold = True
    p_op.font.size = Pt(18)
    p_op.font.color.rgb = BLUE

    open_items = [
        ("Confidence Gate (tau = 50%):", "Evaluates softmax probability. If confidence < 0.50, flags sample as 'Unknown Generator'."),
        ("Prevents False Attribution:", "Protects against newly created AI tools not present in training data."),
        ("Practical Safety:", "Provides reliable alerts in real-world forensic deployment.")
    ]
    for h, t in open_items:
        p_pt = tf_op.add_paragraph()
        p_pt.space_before = Pt(10)
        r1 = p_pt.add_run()
        r1.text = f"• {h} "
        r1.font.bold = True
        r1.font.size = Pt(14)
        r1.font.color.rgb = NAVY
        r2 = p_pt.add_run()
        r2.text = t
        r2.font.size = Pt(13.5)
        r2.font.color.rgb = DARK

    # =============================================================
    # SLIDE 11: CONCLUSION AND FUTURE WORK
    # =============================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11, LIGHT_BG)
    add_header(s11, "Conclusion & Future Work", "SUMMARY & ROADMAP")

    # Left: Conclusion
    add_card(s11, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_c = s11.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Conclusions:"
    p.font.bold = True
    p.font.size = Pt(18)
    p.font.color.rgb = NAVY

    conc_list = [
        "Multi-domain feature fusion outperforms ResNet-18 by +6.5% on detection and +15.6% on attribution.",
        "DINOv2 provides robust semantic embeddings resistant to generator shifts.",
        "Pixel noise alone fails, but adds vital forensic value when combined with style and FFT.",
        "Open-set attribution enables safe real-world deployment."
    ]
    for c in conc_list:
        p_c = tf_c.add_paragraph()
        p_c.text = f"✔  {c}"
        p_c.font.size = Pt(13.5)
        p_c.font.color.rgb = DARK
        p_c.space_before = Pt(10)

    # Right: Future Work
    add_card(s11, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.2))
    tb_fw = s11.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.7))
    tf_fw = tb_fw.text_frame
    tf_fw.word_wrap = True

    p_fw = tf_fw.paragraphs[0]
    p_fw.text = "Future Work:"
    p_fw.font.bold = True
    p_fw.font.size = Pt(18)
    p_fw.font.color.rgb = BLUE

    fw_list = [
        ("Dynamic Attention Gating:", "Adaptively weigh feature branches based on image quality."),
        ("Social Media Compression:", "Harden model against JPEG compression (WhatsApp, X/Twitter)."),
        ("Video Deepfake Extension:", "Extend multi-branch extractors to video frame sequences."),
        ("Visual Heatmap Explanations:", "Add localization heatmaps to highlight exact fake regions.")
    ]
    for h, t in fw_list:
        p_item = tf_fw.add_paragraph()
        p_item.space_before = Pt(10)
        r1 = p_item.add_run()
        r1.text = f"➜ {h} "
        r1.font.bold = True
        r1.font.size = Pt(14)
        r1.font.color.rgb = NAVY
        r2 = p_item.add_run()
        r2.text = t
        r2.font.size = Pt(13)
        r2.font.color.rgb = DARK

    # =============================================================
    # SLIDE 12: REFERENCES
    # =============================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12, LIGHT_BG)
    add_header(s12, "Key References", "BIBLIOGRAPHY")

    add_card(s12, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.2))
    tb_ref = s12.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(4.7))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True

    ref_items = [
        "1. Wang et al. (CVPR 2020) - 'CNN-generated images are surprisingly easy to spot... for now.'",
        "2. Oquab et al. (2023) - 'DINOv2: Learning Robust Visual Features without Supervision.'",
        "3. Frank et al. (ICML 2020) - 'Leveraging Frequency Analysis for Deep Fake Image Recognition.'",
        "4. Bird & Lotfi (IEEE Access 2023) - 'CIFAKE: Image Classification and Explainable Identification of AI Images.'",
        "5. Zhu et al. (NeurIPS 2023) - 'GenImage: A Large-Scale Dataset for AI Image Detection and Attribution.'"
    ]
    for idx, r in enumerate(ref_items):
        p_r = tf_ref.paragraphs[0] if idx == 0 else tf_ref.add_paragraph()
        p_r.text = r
        p_r.font.size = Pt(14)
        p_r.font.color.rgb = DARK
        if idx > 0:
            p_r.space_before = Pt(14)

    # =============================================================
    # SLIDE 13: THANK YOU & Q&A
    # =============================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13, NAVY)

    tb13 = s13.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.33), Inches(3.2))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    p_ty = tf13.paragraphs[0]
    p_ty.text = "Thank You!"
    p_ty.font.bold = True
    p_ty.font.size = Pt(46)
    p_ty.font.color.rgb = WHITE
    p_ty.alignment = PP_ALIGN.CENTER

    p_qa = tf13.add_paragraph()
    p_qa.text = "Questions & Answers"
    p_qa.font.size = Pt(24)
    p_qa.font.color.rgb = RGBColor(147, 197, 253)
    p_qa.alignment = PP_ALIGN.CENTER
    p_qa.space_before = Pt(12)

    p_proj = tf13.add_paragraph()
    p_proj.text = "Major Project: Multi-Domain Forensic Feature Fusion for AI Image Detection"
    p_proj.font.size = Pt(15)
    p_proj.font.color.rgb = RGBColor(203, 213, 225)
    p_proj.alignment = PP_ALIGN.CENTER
    p_proj.space_before = Pt(14)

    prs.save(output_path)
    print(f"Presentation regenerated successfully at: {output_path}")

if __name__ == "__main__":
    create_deck()
