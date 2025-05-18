# Application Documentation

## Run Locally

To run this application locally, follow these steps:

1. Create a Python virtual environment:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create the required directory structure:
   ```bash
   mkdir -p ignored
   ```

5. Place the required ROM files in the ignored directory:
   - Copy `dd.gb` to `ignored/dd.gb`
   - Copy `dd.gb.state` to `ignored/dd.gb.state`

6. Run the application:
   ```bash
   python3 run-ray-train.py
   ```

Note: Make sure you have Python 3.11 installed as this version is required for compatibility with all dependencies.

## Building the Container Image

### Prerequisites

1. OpenShift Cluster with Tekton installed
2. `oc` CLI tool installed and logged into the cluster
3. Access to build and push images in your namespace

### Setting Up the Build Environment

1. Create the required Kubernetes resources:
   ```bash
   # Create the Service Account and PVC
   oc apply -f .tkn/hello-chris-sa.yaml
   oc apply -f .tkn/pvc.yaml

   # Apply Tekton Tasks
   oc apply -f .tkn/buildah-task.yaml
   oc apply -f .tkn/git-clone-task.yaml
   ```

2. Set up the RPM serving infrastructure:
   ```bash
   # Create an upload pod for temporary file storage
   oc create -f - <<EOF
   apiVersion: v1
   kind: Pod
   metadata:
     name: upload-pod
   spec:
     containers:
     - name: upload-pod
       image: nginx
       ports:
       - containerPort: 80
     volumes:
     - name: shared-data
       emptyDir: {}
   EOF

   # Wait for the pod to be ready
   oc wait --for=condition=ready pod/upload-pod

   # Copy the RPM file to the pod
   oc cp path/to/your.rpm upload-pod:/usr/share/nginx/html/

   # Create a service to expose the nginx server
   oc expose pod upload-pod --port=80
   ```

3. Configure the pipeline:
   ```bash
   # Apply the pipeline definition
   oc apply -f .tkn/hello-chris-pipeline.yaml
   ```

### Running the Build

1. Start the pipeline:
   ```bash
   oc apply -f .tkn/run.yaml
   ```

2. Monitor the build progress:
   ```bash
   tkn pipelinerun logs -f
   ```

### Pipeline Structure

The Tekton pipeline (.tkn/hello-chris-pipeline.yaml) consists of the following stages:
- Git clone of the source repository
- Building the container image using Buildah
- Pushing the image to the specified registry

### Build Assets

The `.tkn` directory contains all necessary Tekton resources:
- `buildah-task.yaml`: Task definition for building container images
- `git-clone-task.yaml`: Task for cloning Git repositories
- `hello-chris-pipeline.yaml`: Main pipeline definition
- `hello-chris-sa.yaml`: Service Account configuration
- `pvc.yaml`: Persistent Volume Claim for build workspace
- `run.yaml`: PipelineRun definition to trigger the build

### Image Build Process

The image is built using a multi-stage Dockerfile located in the `image/` directory. It:
1. Installs system dependencies
2. Sets up Python environment
3. Installs Python packages from Pipfile.lock
4. Configures the runtime environment for the Ray application

Note: The build process requires access to both PyPI and the temporary nginx server hosting the RPM file.

