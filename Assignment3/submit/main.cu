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
		translations.push_back (command);
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
__device__ void DFS(int source, int* hOffsetGpu, int* hCsrGpu, bool* visited, int *hGlobalCoordinatesXGpu, int *hGlobalCoordinatesYGpu, int *hTranslationGpu) {
    visited[source] = true;
    // printf("%d ", vertex);
    
    int c = hTranslationGpu[(source)*3 + 1];
    int a = hTranslationGpu[(source)*3 + 2];

    if(c==0){
      atomicAdd(&hGlobalCoordinatesYGpu[source], a);
    }
    if(c==1){
      atomicAdd(&hGlobalCoordinatesYGpu[source], -a);
    }
    if(c==2){
      atomicAdd(&hGlobalCoordinatesXGpu[source], -a);
    }
    if(c==3){
      atomicAdd(&hGlobalCoordinatesXGpu[source], a);
    }

    int start = hOffsetGpu[source];
    int end = hOffsetGpu[source + 1];

    for (int i = start; i < end; ++i) {
      int neighbor = hCsrGpu[i];
      if (!visited[neighbor]) {
        DFS(neighbor, hOffsetGpu, hCsrGpu, visited, hGlobalCoordinatesXGpu, hGlobalCoordinatesYGpu, hTranslationGpu);
      }
    }
}
__global__ void translationOperation(int *hCsrGpu, int *hOffsetGpu, int *hGlobalCoordinatesXGpu, int *hGlobalCoordinatesYGpu, int *hTranslationGpu, int V){
  int global_Index = blockIdx.x * blockDim.x + threadIdx.x;
  int n = hTranslationGpu[global_Index*3];
  bool visited[1000] = {false};
  DFS(n, hOffsetGpu, hCsrGpu, visited, hGlobalCoordinatesXGpu, hGlobalCoordinatesYGpu, hTranslationGpu);
  
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
	for(int i = 0; i < V+1; i++){
    printf("%d ", hOffset[i]);
	}
	printf("\n");
	for(int i = 0; i < E; i++){
		printf("%d ", hCsr[i]);
	}
  printf("hello");
  printf("hellobhatia");
  printf("helloyashn\n");
	printf("\n");
  int *hOffsetGpu;
  int *hCsrGpu;
  int *hOpacityGpu;
  int **hMeshGpu;
  int *hGlobalCoordinatesXGpu;
  int *hGlobalCoordinatesYGpu;
  int *hFrameSizeXGpu;
  int *hFrameSizeYGpu;
  int *hTranslationGpu;
  
  cudaMalloc(&hOffsetGpu, (V+1)*sizeof(int));
	cudaMemcpy(hOffsetGpu,hOffset,(V+1)*sizeof(int),cudaMemcpyHostToDevice);
		
	cudaMalloc(&hCsrGpu, (E)*sizeof(int));
	cudaMemcpy(hCsrGpu,hCsr,(E)*sizeof(int),cudaMemcpyHostToDevice);
		
	cudaMalloc(&hOpacityGpu, (V)*sizeof(int));
	cudaMemcpy(hOpacityGpu,hOpacity,(V)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMalloc(&hMeshGpu, V * sizeof(int*));

  for (int i = 0; i < V; i++) {
    int *dPtr;
    cudaMalloc(&dPtr, hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int));
    cudaMemcpy(dPtr, hMesh[i], hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&hMeshGpu[i], &dPtr, sizeof(int*), cudaMemcpyHostToDevice);
  }

  
	cudaMalloc(&hGlobalCoordinatesXGpu, (V)*sizeof(int));
	cudaMemcpy(hGlobalCoordinatesXGpu,hGlobalCoordinatesX,(V)*sizeof(int),cudaMemcpyHostToDevice);
		
	cudaMalloc(&hGlobalCoordinatesYGpu, (V)*sizeof(int));
	cudaMemcpy(hGlobalCoordinatesYGpu,hGlobalCoordinatesY,(V)*sizeof(int),cudaMemcpyHostToDevice);
		
	cudaMalloc(&hFrameSizeXGpu, (V)*sizeof(int));
	cudaMemcpy(hFrameSizeXGpu,hFrameSizeX,(V)*sizeof(int),cudaMemcpyHostToDevice);
		
	cudaMalloc(&hFrameSizeYGpu, (V)*sizeof(int));
	cudaMemcpy(hFrameSizeYGpu,hFrameSizeY,(V)*sizeof(int),cudaMemcpyHostToDevice);
	
  printf("helloiiiii");
  
cudaMalloc(&hTranslationGpu, 3*(V)*sizeof(int));

cudaMemcpy(&hTranslationGpu, translations.data(), 3*(numTranslations)*sizeof(int),cudaMemcpyHostToDevice);
	// Do not change anything below this comment.
	// Code ends here.
  printf("helloiiiii");

int noBlock = numTranslations/1024;
translationOperation<<<noBlock, 1024>>>(hCsrGpu, hOffsetGpu, hGlobalCoordinatesXGpu, hGlobalCoordinatesYGpu, hTranslationGpu);



	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
