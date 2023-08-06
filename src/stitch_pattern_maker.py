from PIL import Image, ImageDraw

# line color = (0, 0, 255)

def draw_graph_paper_on_image(image, grid_size, line_color=(0, 0, 0), line_thickness=2):
    # Open the image using PIL
    width, height = image.size

    # Create a blank image with the same size as the input image
    graph_paper = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(graph_paper)

    # Draw vertical lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_thickness)
        sub_part_size = grid_size // 10
        for i in range(1, 10):
            draw.line([(x + i * sub_part_size, 0), (x + i * sub_part_size, height)], fill=line_color, width=1)

    # Draw horizontal lines
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=line_color, width=line_thickness)
        sub_part_size = grid_size // 10
        for i in range(1, 10):
            draw.line([(0, y + i * sub_part_size), (width, y + i * sub_part_size)], fill=line_color, width=1)
    # Overlay the graph paper on the original image
    result = Image.alpha_composite(image, graph_paper)

    return result

def stitch_pattern(quantized_image, stitch_size=10, stitch_width=10):

    for y in range(0, quantized_image.height, stitch_size):
        for x in range(0, quantized_image.width, stitch_size):
            cross_color = quantized_image.getpixel((x, y))
            
            # Create a new image with a white background for the 10x10 set of pixels
            set_image = Image.new("RGB", (stitch_size, stitch_size), (255, 255, 255))
            
            # Create a draw object for the set image
            draw = ImageDraw.Draw(set_image)
            
            # Draw a cross on the set image
            draw.line([(0, 0), (stitch_size-1, stitch_size-1)], fill=cross_color, width=stitch_width)
            draw.line([(0, stitch_size-1), (stitch_size-1, 0)], fill=cross_color, width=stitch_width)
            
            # Paste the set image onto the graph paper at the specified position
            quantized_image.paste(set_image, (x, y))
    
        # Convert the image to RGBA mode
    quantized_image = quantized_image.convert("RGBA")
    # Create the graph paper with the specified grid size
    graph_paper = draw_graph_paper_on_image(quantized_image, grid_size=stitch_size*10)
    graph_paper = graph_paper.convert('RGB')

    return graph_paper

def save_png_as_pdf(png_path, pdf_path):
    # Open the PNG image
    image = Image.open(png_path)

    # Create a new blank PDF image with the same size
    pdf_image = Image.new('RGB', image.size, 'white')

    # Paste the PNG image onto the PDF image
    pdf_image.paste(image, (0, 0))

    # Save the PDF image
    pdf_image.save(pdf_path, 'PDF', resolution=100.0)

from matplotlib.backends.backend_pdf import PdfPages

def save_png_as_pdf_matplotlib(png_path, pdf_path):
    # Open the PNG image
    image = plt.imread(png_path)

    # Create a new PDF file
    with PdfPages(pdf_path) as pdf:
        # Create a new figure and plot the image
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')

        # Save the figure to the PDF file
        pdf.savefig(fig)

        # Close the figure
        plt.close(fig)