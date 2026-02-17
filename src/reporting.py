import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from src.viz import plot_sales_trends_static, plot_top_categories_static

def create_ppt_presentation(df, insights, recommendations):
    """
    Generates a PowerPoint presentation based on the current data and insights.
    
    Args:
        df (pd.DataFrame): The main dataset (cleaned/featured).
        insights (dict): Dictionary of calculated insights (elasticity, promo uplift).
        recommendations (list): List of recommendation dictionaries.
        
    Returns:
        BytesIO: A binary stream containing the PPTX file.
    """
    # Load template if exists
    template_path = "assets/Ref_Template.pptx"
    if not os.path.exists(template_path):
        template_path = "data/Ref_Template.pptx"

    if os.path.exists(template_path):
        prs = Presentation(template_path)
    else:
        prs = Presentation()
    
    # Helper to add a title slide
    def add_title_slide(title, subtitle):
        # Use layout 0 if default, or try to be smart if template
        layout_idx = 0 
        slide_layout = prs.slide_layouts[layout_idx] 
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        if len(slide.placeholders) > 1:
            slide.placeholders[1].text = subtitle

    # Helper to add a content slide with image
    def add_chart_slide(title, fig, commentary):
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = title
        
        # Save plot to stream
        img_stream = io.BytesIO()
        fig.savefig(img_stream, format='png', bbox_inches='tight', dpi=150)
        img_stream.seek(0)
        
        # Add image
        # Left, Top, Width
        slide.shapes.add_picture(img_stream, Inches(0.5), Inches(1.5), width=Inches(6), height=Inches(4))
        
        # Add commentary text box
        # Enable text wrapping!
        txBox = slide.shapes.add_textbox(Inches(6.6), Inches(1.5), Inches(3.2), Inches(4))
        tf = txBox.text_frame
        tf.word_wrap = True # Fix for overflow
        
        p = tf.add_paragraph()
        p.text = "Key Insights:"
        p.font.bold = True
        
        p = tf.add_paragraph()
        p.text = commentary
        p.font.size = Pt(14)

    # 1. Title Slide
    add_title_slide("Retail Strategic Analysis", f"Generated on {pd.Timestamp.now().date()}\nAutomated Pricing & Forecasting Report")

    # 2. Executive Summary Slide
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Executive Summary"
    content = slide.placeholders[1].text_frame
    content.text = "Analysis Overview"
    
    p = content.add_paragraph()
    p.text = f"Dataset covers {len(df):,} transactions."
    p.level = 1
    p = content.add_paragraph()
    p.text = f"Analyzed {df['Category'].nunique()} Categories and {df['Store ID'].nunique() if 'Store ID' in df else df['Store'].nunique()} Stores."
    p.level = 1
    
    if 'top_inelastic' in insights:
        top_cat = insights['top_inelastic'].iloc[0]['Category']
        p = content.add_paragraph()
        p.text = f"Top Opportunity: '{top_cat}' shows low price sensitivity (Inelastic)."
        p.level = 1
        p.font.bold = True

    # 3. Sales Trends Slide
    fig_trend = plot_sales_trends_static(df, title="Sales Performance Over Time")
    add_chart_slide("Sales Trends", fig_trend, "The chart illustrates seasonal peaks and overall volume trajectory. Analyze December spikes for inventory planning.")
    plt.close(fig_trend)

    # 4. Top Categories Slide
    if 'Category' in df.columns:
        fig_cat = plot_top_categories_static(df, cat_col='Category', value_col='Units Sold', title="Top Categories by Volume")
        add_chart_slide("Category Performance", fig_cat, "Identify the core revenue drivers. Ensure stock availability for these top performers.")
        plt.close(fig_cat)

    # 5. Elasticity Slide
    if 'elastic_df' in insights and not insights['elastic_df'].empty:
        slide = prs.slides.add_slide(prs.slide_layouts[1]) # Title and Content
        slide.shapes.title.text = "Price Elasticity Analysis"
        
        # Create a table
        rows = 5
        cols = 3
        table_left = Inches(1)
        table_top = Inches(2)
        table_width = Inches(8)
        table_height = Inches(0.8)
        
        shape = slide.shapes.add_table(rows, cols, table_left, table_top, table_width, table_height)
        table = shape.table
        
        # Header
        table.cell(0, 0).text = "Category"
        table.cell(0, 1).text = "Elasticity"
        table.cell(0, 2).text = "Classification"
        
        # Fill
        top_res = insights['elastic_df'].sort_values('Elasticity').head(4) # Most elastic negative
        for i, row in enumerate(top_res.iterrows()):
             idx, data = row
             r = i + 1
             table.cell(r, 0).text = str(data['Category'])
             table.cell(r, 1).text = f"{data['Elasticity']:.2f}"
             classification = "Elastic (Sensitive)" if data['Elasticity'] < -1 else "Inelastic (Sticky)"
             table.cell(r, 2).text = classification

    # 6. Strategic Recommendations
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Strategic Recommendations"
    content = slide.placeholders[1].text_frame
    content.text = "Proposed Actions based on Data:"
    
    for rec in recommendations[:4]: # Limit to fit slide
        p = content.add_paragraph()
        p.text = f"• {rec['Action']}"
        p.font.bold = True
        
        sub = content.add_paragraph()
        sub.text = f"   Rationale: {rec['Rationale']}"
        sub.font.size = Pt(14)
        sub.level = 1

    # Save to buffer
    ppt_stream = io.BytesIO()
    prs.save(ppt_stream)
    ppt_stream.seek(0)
    
    return ppt_stream

