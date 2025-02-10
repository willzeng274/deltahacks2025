import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import os
import sys
import numpy as np

import tempfile


class FaceRecognitionSystem:
    def __init__(self, embeddings=None):
        # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print("Using device:", device)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        cascade_file = self._get_cascade_file()
        if cascade_file is None:
            raise Exception("Could not find the cascade file")

        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        # Load model
        self.model = models.resnet18(pretrained=True)
        self.model.to(device)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.embeddings = embeddings or {}

        # self.embeddings_dir = Path("embeddings")
        # self.embeddings_dir.mkdir(exist_ok=True)

        # self.embeddings = self.load_all_embeddings()

    def identify_person(self, uploaded_file: bytes, threshold=0.6):
        if not self.embeddings:
            return "Unknown", 0.0

        # Fast image loading
        file_bytes = np.frombuffer(uploaded_file, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("Invalid image")

        # Resize image before processing to reduce computation
        height, width = image.shape[:2]
        max_dim = 640  # Limit maximum dimension
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Convert to grayscale and detect faces with optimized parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return "Unknown", 0.0

        # Process only the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract and process face
        face = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face).unsqueeze(0)

        # Get pre-computed stored embeddings
        stored_names = list(self.embeddings.keys())
        stored_embeddings = torch.stack([self.embeddings[name] for name in stored_names])

        # Generate embedding with no_grad for speed
        with torch.no_grad():
            embedding = self.model(face_tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            # Fast similarity computation
            similarities = torch.cosine_similarity(embedding, stored_embeddings)
            best_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_idx].item()

        if best_similarity > threshold:
            return stored_names[best_idx], best_similarity
        print("Best similarity is: ", best_similarity)
        return "Unknown", 0.0

    def _get_cascade_file(self):
        possible_paths = [
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            os.path.join(
                sys.prefix,
                "share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            ),
            "haarcascade_frontalface_default.xml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        try:
            import urllib.request

            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            return "haarcascade_frontalface_default.xml"
        except:
            return None

    def process_uploaded_video(self, file: bytes, person_name: str):
        # Write bytes to temporary file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            temp.write(file)
            temp.close()
            
            frames = []
            frame_count = 0
            video_capture = cv2.VideoCapture(temp.name)
            if not video_capture.isOpened():
                raise Exception("Can't open the uploaded video. Are you sure it's valid?")

            iterations = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Only process every 10th frame
                if frame_count % 10 == 0:
                    iterations += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    if len(faces) == 1:
                        x, y, w, h = faces[0]
                        face = frame_rgb[y : y + h, x : x + w]
                        frames.append(face)

                frame_count += 1

            print(f"Processed {iterations} frames from the video.")
            if not frames:
                return None

            return self._process_frames_and_dump(frames, person_name)
        except Exception as e:
            raise e
        finally:
            video_capture.release()
            os.unlink(temp.name)

    def _process_frames_and_dump(self, frames, person_name):
        # Let's process those frames, generate an embedding, and give you a pickle blob.
        embeddings = []
        for face in frames:
            face_tensor = self.transform(face).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = torch.nn.functional.normalize(
                    embedding.squeeze(), p=2, dim=0
                )
                embeddings.append(embedding)

        if embeddings:
            final_embedding = torch.stack(embeddings).median(dim=0)[0]
            data = pickle.dumps(final_embedding)
            return data
        else:
            print("Warning: No embeddings.")

    # use .pkl file to save the embedding
    # def load_all_embeddings(self):
    #     """Load all embeddings from individual files"""
    #     embeddings = {}
    #     if self.embeddings_dir.exists():
    #         for embedding_file in self.embeddings_dir.glob("*.pkl"):
    #             name = embedding_file.stem  # Get name without .pkl extension
    #             with open(embedding_file, "rb") as f:
    #                 embeddings[name] = pickle.load(f)
    #     return embeddings
    #
    # def save_embedding(self, name, embedding):
    #     """Save embedding to individual file"""
    #     embedding_file = self.embeddings_dir / f"{name}.pkl"
    #     with open(embedding_file, "wb") as f:
    #         pickle.dump(embedding, f)
    #     # Update in-memory embeddings
    #     self.embeddings[name] = embedding

    # use .pt file to save the embedding
    # def save_embedding(self, name, embedding):
    #     """Save embedding with torch.save"""
    #     embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)  # Normalize when saving
    #     print("The embedding is: ", embedding)
    #     embedding_file = self.embeddings_dir / f"{name}.pt"
    #     torch.save(embedding, embedding_file)
    #     # Update in-memory embeddings
    #     self.embeddings[name] = embedding

    # def load_all_embeddings(self):
    #     """Load all embeddings using torch.load"""
    #     embeddings = {}
    #     if self.embeddings_dir.exists():
    #         for embedding_file in self.embeddings_dir.glob('*.pt'):
    #             name = embedding_file.stem
    #             print(f"Loading embedding for {name}...")
    #             embedding = torch.load(embedding_file)
    #             embeddings[name] = embedding
    #             print("The loaded embedding is: ", embedding)
    #             print(f"Loaded embedding for {name}, shape: {embedding.shape}")
    #     return embeddings

    # def record_video(self, person_name, duration=5, fps=20):
    #     """Record video with proper cleanup"""
    #     cap = cv2.VideoCapture(0)
    #     if not cap.isOpened():
    #         raise Exception("Could not open camera")

    #     start_time = time.time()
    #     frames = []

    #     print(f"Recording {duration} seconds of video for {person_name}...")

    #     while time.time() - start_time < duration:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = self.face_cascade.detectMultiScale(
    #             gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    #         )

    #         if len(faces) == 1:  # Only process frames with exactly one face
    #             x, y, w, h = faces[0]
    #             face = frame_rgb[y : y + h, x : x + w]
    #             frames.append(face)
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #         cv2.imshow("Recording...", frame)
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break

    #     # Proper cleanup
    #     print("Stopping camera...")
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)  # Extra waitKey calls to ensure windows close
    #     cv2.waitKey(1)
    #     cv2.waitKey(1)
    #     cv2.waitKey(1)

    #     if not frames:
    #         print("No faces detected in the video")
    #         return

    #     # Process frames and generate embedding
    #     embeddings = []
    #     for face in frames:
    #         face_tensor = self.transform(face)
    #         face_tensor = face_tensor.unsqueeze(0)

    #         with torch.no_grad():
    #             embedding = self.model(face_tensor)
    #             embedding = torch.nn.functional.normalize(
    #                 embedding.squeeze(), p=2, dim=0
    #             )
    #             embeddings.append(embedding)

    #     if embeddings:
    #         # Use median for robustness
    #         final_embedding = torch.stack(embeddings).median(dim=0)[0]
    #         self.save_embedding(person_name, final_embedding)
    #         print(f"Saved embedding for {person_name}")
    #         # Print number of stored embeddings
    #         # print(f"Total people stored: {len(self.embeddings)}")
    #         print("Stored names:", list(self.embeddings.keys()))

    

    # def identify_person(self, image_path, threshold=0.6):
    #     """Identify person with debug information"""
    #     if not self.embeddings:
    #         print("No embeddings stored yet! Please record some faces first.")
    #         return []

    #     image = cv2.imread(image_path)
    #     if image is None:
    #         raise Exception(f"Could not read image: {image_path}")

    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     faces = self.face_cascade.detectMultiScale(
    #         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    #     )

    #     results = []
    #     for (x, y, w, h) in faces:
    #         face = image_rgb[y:y+h, x:x+w]
    #         face_tensor = self.transform(face)
    #         face_tensor = face_tensor.unsqueeze(0)

    #         with torch.no_grad():
    #             embedding = self.model(face_tensor)
    #             embedding = torch.nn.functional.normalize(embedding.squeeze(), p=2, dim=0)

    #         best_match = None
    #         best_similarity = -1

    #         # Debug information
    #         print("\nStored embeddings:")
    #         for name, stored_embedding in self.embeddings.items():
    #             print("Stored embbeding is: ", stored_embedding)
    #             print("Target embedding is: ", embedding)
    #             similarity = torch.cosine_similarity(
    #                 embedding.unsqueeze(0),
    #                 stored_embedding.unsqueeze(0)
    #             ).item()

    #             print(f"Comparing with {name}: similarity = {similarity:.3f}")

    #             if similarity > best_similarity:
    #                 best_similarity = similarity
    #                 best_match = name

    #         if best_similarity > threshold:
    #             results.append((best_match, best_similarity))
    #         else:
    #             results.append(("Unknown", best_similarity))

    #     return results


if __name__ == "__main__":
    system = FaceRecognitionSystem()

    while True:
        print("\n1. Record new person")
        print("2. Identify person")
        print("3. List stored people")
        print("4. Exit")
        choice = input("Choose an option (1-4): ")

        if choice == "1":
            person_name = input("Enter person's name: ")
            system.record_video(person_name)
        elif choice == "2":
            if not system.embeddings:
                print("No embeddings stored yet! Please record some faces first.")
                continue
            test_image = input("Enter path to test image: ")
            results = system.identify_person(test_image)
            for person, confidence in results:
                print(f"Identified: {person} (confidence: {confidence:.2f})")
        elif choice == "3":
            print("\nStored people:", list(system.embeddings.keys()))
        elif choice == "4":
            break
