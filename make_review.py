from fpdf import FPDF


def make_pdf_review():
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="crimes in boston", ln=1, align="C")
    pdf.image('./static/opt_clusters_num.png', x=10, y=8, w=100)
    pdf.image('./static/district_clustering.png', x=10, y=500, w=100)
    pdf.image('./static/district_and_offence_code_clustering.png', x=10, y=1000, w=100)
    pdf.image('./static/hierarchy_clusters.png', x=10, y=1400, w=100)
    pdf.image('./static/hierarchy_clustering.png', x=10, y=1400, w=100)
    pdf.output("./reviews/review.pdf")

if __name__ == '__main__':
    make_pdf_review()
