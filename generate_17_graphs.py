import sys
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import predict from app.py
from app import predict

def main():
    # Find a test image
    test_dir = 'Dataset/Testing/glioma'
    test_img_path = os.path.join(test_dir, os.listdir(test_dir)[0])
    print(f"Using test image: {test_img_path}")
    
    # Load image as np array
    img = np.array(Image.open(test_img_path).convert('RGB'))
    
    # Run prediction
    results = predict(img)
    
    # Unpack
    (
        label_dict, cam_image, fig_clahe, fig_radar, fig_hist, 
        fig_feat, fig_saliency, fig_3d, fig_gauge, fig_guided, 
        fig_var, fig_contour, fig_channels, fig_sev, fig_edges, 
        fig_profile, fig_water, fig_saliency_ratio, clinical_report, status_html
    ) = results
    
    # Save outputs
    artifact_dir = r"C:\Users\MUTHURAMANRAMANATHAN\.gemini\antigravity\brain\4d6ffc33-742e-4e94-8526-190c9c0242bb"
    
    # 1. CAM Image
    Image.fromarray(cam_image).save(os.path.join(artifact_dir, "graph_1_cam.png"))
    
    figures = {
        "graph_2_clahe.png": fig_clahe,
        "graph_3_radar.png": fig_radar,
        "graph_4_hist.png": fig_hist,
        "graph_5_feat.png": fig_feat,
        "graph_6_saliency.png": fig_saliency,
        "graph_7_3d.png": fig_3d,
        "graph_8_gauge.png": fig_gauge,
        "graph_9_guided.png": fig_guided,
        "graph_10_var.png": fig_var,
        "graph_11_contour.png": fig_contour,
        "graph_12_channels.png": fig_channels,
        "graph_13_sev.png": fig_sev,
        "graph_14_edges.png": fig_edges,
        "graph_15_profile.png": fig_profile,
        "graph_16_water.png": fig_water,
        "graph_17_saliency_ratio.png": fig_saliency_ratio
    }
    
    for filename, fig in figures.items():
        if isinstance(fig, plt.Figure):
            fig.savefig(os.path.join(artifact_dir, filename), bbox_inches='tight')
            plt.close(fig)
        else:
            print(f"Warning: {filename} is not a plt.Figure, it is {type(fig)}")
            
    print("Successfully generated and saved all 17 graphs!")

if __name__ == "__main__":
    main()
