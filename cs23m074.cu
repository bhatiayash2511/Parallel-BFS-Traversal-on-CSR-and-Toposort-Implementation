
/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

__device__ void dfs(int startNode, int C, int A, int *deviceGlobalCoordinatesX, int *deviceGlobalCoordinatesY, int* deviceOffset, int* deviceCsr, bool *visited, int V) {
    const int MAX_QUEUE_SIZE = 1e7; // Adjust the queue size as needed
    int queue[MAX_QUEUE_SIZE];
    int front = 0; // Front of the queue
    int rear = -1; // Rear of the queue

    // Enqueue the start node
    queue[++rear] = startNode;

    while (front <= rear) {
        // Dequeue a node from the front of the queue
        int currentNode = queue[front++];

        // If the node has not been visited yet
        if (!visited[currentNode]) {
            visited[currentNode] = true;

            // Perform the atomic operation based on the condition C
            if (C == 0) {
                atomicSub(&deviceGlobalCoordinatesX[currentNode], A);
            } else if (C == 1) {
                atomicAdd(&deviceGlobalCoordinatesX[currentNode], A);
            } else if (C == 2) {
                atomicSub(&deviceGlobalCoordinatesY[currentNode], A);
            } else {
                atomicAdd(&deviceGlobalCoordinatesY[currentNode], A);
            }

            // Iterate over the CSR representation to find neighbors of the current node
            int start = deviceOffset[currentNode];
            int end = deviceOffset[currentNode + 1];
            for (int i = start; i < end; ++i) {
                int neighbor = deviceCsr[i];
                // If the neighbor has not been visited, enqueue it
                if (!visited[neighbor] && rear < MAX_QUEUE_SIZE - 1) {
                    queue[++rear] = neighbor;
                }
            }
        }
    }
}


__global__ void dfsInGraph(int *deviceGlobalCoordinatesX,int *deviceGlobalCoordinatesY,int *deviceOffset,int *deviceCsr,int *deviceTranslations,int V,int numTranslations){


		unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
		if(id<numTranslations){

		int ind=id*3;
		int N=deviceTranslations[ind];
		int C=deviceTranslations[ind+1];
		int A=deviceTranslations[ind+2];

		const int size = 1e7;
		bool visited[size] = {false};
		
		dfs(N,C,A,deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,deviceOffset,deviceCsr,visited,V);
   
		}
}

__global__ void fillFinalGraph(int i,int *deviceArrOpacity,int *deviceOpacity,int *deviceFinalPng,int frameSizeX,int frameSizeY,int deviceGlobalCoordinatesX_i,int deviceGlobalCoordinatesY_i,int deviceFrameSizeX_i,int deviceFrameSizeY_i,int *deviceMesh_i){
 // printf("\nhehe");
	unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<deviceFrameSizeX_i*deviceFrameSizeY_i){
	int row=id/deviceFrameSizeY_i;
	int col=id%deviceFrameSizeY_i;
	int correctedRow=row+deviceGlobalCoordinatesX_i;
	int correctedCol=col+deviceGlobalCoordinatesY_i;
	unsigned index=correctedRow*frameSizeY+correctedCol;
	if(correctedRow>=0 && correctedRow<frameSizeX && correctedCol>=0 && correctedCol<frameSizeY && deviceArrOpacity[index]<deviceOpacity[i]){
		deviceArrOpacity[index]=deviceOpacity[i];
		deviceFinalPng[index]=deviceMesh_i[id];
	}
	}
}

int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;
	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;
	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	
  auto start = std::chrono::high_resolution_clock::now () ;
	// Code begins here.
	// Do not change anything above this comment.


	int *htranslations=new int[translations.size()*3];
	for(int i=0;i<translations.size();i++){
		htranslations[i*3+0]=translations[i][0];
		htranslations[i*3+1]=translations[i][1];
		htranslations[i*3+2]=translations[i][2];
	}

	int *deviceTranslations;
	cudaMalloc(&deviceTranslations, sizeof(int) * numTranslations * 3);
  cudaMemcpy(deviceTranslations , htranslations, sizeof(int) *numTranslations* 3, cudaMemcpyHostToDevice);

	int blockSize=1024;
	int gridSize=(numTranslations+blockSize-1)/blockSize;
	int *deviceGlobalCoordinatesX,*deviceGlobalCoordinatesY;
	cudaMalloc(&deviceGlobalCoordinatesX,sizeof(int)*V);
	cudaMalloc(&deviceGlobalCoordinatesY,sizeof(int)*V);
	cudaMemcpy(deviceGlobalCoordinatesX,hGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyHostToDevice);
	cudaMemcpy(deviceGlobalCoordinatesY,hGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyHostToDevice);

	int *deviceOffset,*deviceCsr;
	cudaMalloc(&deviceOffset,sizeof(int)*(V+1));
	cudaMalloc(&deviceCsr,sizeof(int)*E);
  cudaMemcpy(deviceOffset,hOffset,sizeof(int)*(V+1),cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCsr,hCsr,sizeof(int)*E,cudaMemcpyHostToDevice);

  
	dfsInGraph<<<gridSize,blockSize>>>(deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,deviceOffset,deviceCsr,deviceTranslations,V,numTranslations);

	cudaDeviceSynchronize();
	cudaMemcpy(hGlobalCoordinatesX,deviceGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyDeviceToHost);
	cudaMemcpy(hGlobalCoordinatesY,deviceGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyDeviceToHost);
	

/*	printf("printing the coordinates\n");
	for(int i=0;i<V;i++){
		printf("%d %d \n",hGlobalCoordinatesX[i],hGlobalCoordinatesY[i]);
	}*/
	
	//writing on the final scene
	int *hArrOpacity = (int *)malloc(frameSizeX * frameSizeY * sizeof(int));

	
	long long s=frameSizeX * frameSizeY;
    // Initialize the array elements to -1
  for (long long i = 0; i <s ; i++) {

        hArrOpacity[i] = -1;
    }


	int *deviceArrOpacity;
	cudaMalloc(&deviceArrOpacity,sizeof(int)*frameSizeX*frameSizeY);
	cudaMemcpy(deviceArrOpacity,hArrOpacity,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyHostToDevice);

	int *deviceFinalPng;
	cudaMalloc(&deviceFinalPng,sizeof(int)*frameSizeX*frameSizeY);
	cudaMemcpy(deviceFinalPng,hFinalPng,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyHostToDevice);



	int *deviceFrameSizeX;// = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *deviceFrameSizeY;// = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	cudaMalloc(&deviceFrameSizeX,sizeof(int)*V);
	cudaMalloc(&deviceFrameSizeY,sizeof(int)*V);


	cudaMemcpy(deviceFrameSizeX,hFrameSizeX,sizeof(int)*V,cudaMemcpyHostToDevice);
	cudaMemcpy(deviceFrameSizeY,hFrameSizeY,sizeof(int)*V,cudaMemcpyHostToDevice);

int *deviceOpacity;
cudaMalloc(&deviceOpacity,sizeof(int)*V);
cudaMemcpy(deviceOpacity,hOpacity,sizeof(int)*V,cudaMemcpyHostToDevice);





	for(long long i=0;i<V;i++){

		int Xsize=hFrameSizeX[i];
		int Ysize=hFrameSizeY[i];
		int meshSize=Xsize*Ysize;
		int blockSize=100;
		int gridSize=(meshSize+blockSize-1)/blockSize;
		int *mesh_i=hMesh[i];
		int *deviceMesh_i;
		cudaMalloc(&deviceMesh_i,sizeof(int)*meshSize);
		cudaMemcpy(deviceMesh_i,mesh_i,sizeof(int)*meshSize,cudaMemcpyHostToDevice);
		fillFinalGraph<<<gridSize,blockSize>>>(i,deviceArrOpacity,deviceOpacity,deviceFinalPng,frameSizeX,frameSizeY,hGlobalCoordinatesX[i],hGlobalCoordinatesY[i],hFrameSizeX[i],hFrameSizeY[i],deviceMesh_i);
		cudaMemcpy(hFinalPng,deviceFinalPng,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}



	cudaDeviceSynchronize();
	/*
	printf("\n");
	for(int i=0;i<frameSizeX;i++){
		for(int j=0;j<frameSizeY;j++){
			printf("%d ",hFinalPng[i*frameSizeY+j]);
		}
		printf("\n");
	}
	*/
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	//printf ("execution time : %f\n",(double)timeTaken) ;
	printf("execution time: %f\n", static_cast<double>(timeTaken.count()));

	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;
}