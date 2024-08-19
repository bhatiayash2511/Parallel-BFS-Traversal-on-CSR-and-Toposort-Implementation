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
#include<iostream>


  void __global__  calculatingFinalImage(int *gpu_hFinalPng,int frameSizeX,int frameSizeY,int *gpu_hashOpacity,int *gpu_hGlobalCoordinatesX,int *gpu_hGlobalCoordinatesY,int *gpu_hFrameSizeY, int **gpu_hMesh){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<frameSizeX*frameSizeY){
      if(gpu_hFinalPng[idx]!=-1){
        int mesh_id = gpu_hashOpacity[gpu_hFinalPng[idx]];
      int row = idx/frameSizeY;
      int col = idx%frameSizeY;
      int meshX = row - gpu_hGlobalCoordinatesX[mesh_id];
      int meshY = col - gpu_hGlobalCoordinatesY[mesh_id];
      int sizeY = gpu_hFrameSizeY[mesh_id];
      gpu_hFinalPng[idx] = gpu_hMesh[mesh_id][meshX*sizeY+meshY];
      }
      else{
        gpu_hFinalPng[idx] =0;
      }
    }
  }


void __global__ hashingOpacity(int* gpu_hOpacity,int *gpu_hashOpacity,int nodes){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx<nodes){
    gpu_hashOpacity[gpu_hOpacity[idx]] = idx;
  }
}

void __global__ fixingOpacity(int numberOfMeshes,int *gpu_hFinalPng, int frameSizeX, int frameSizeY, int * gpu_hOpacity, int *gpu_hFrameSizeX, int *gpu_hFrameSizeY, int *gpu_hGlobalCoordinatesX, int *gpu_hGlobalCoordinatesY){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<numberOfMeshes){
    int sizex = gpu_hFrameSizeX[idx];
    int sizey = gpu_hFrameSizeY[idx];
    int globalX = gpu_hGlobalCoordinatesX[idx];
    int globalY = gpu_hGlobalCoordinatesY[idx];
    for(int i =0;i<sizex;i++){
      int x = i+globalX;
      for(int j=0;j<sizey;j++){
        int y = j+globalY;
        if(x>=0 && y>=0 && x<frameSizeX && y<frameSizeY){
          atomicMax(&gpu_hFinalPng[x*frameSizeY+y],gpu_hOpacity[idx]);
        }
      }
    }
  }
}

void __global__ calculatingTranslations(int numTranslations,int *gpu_hGlobalCoordinatesX,int *gpu_hGlobalCoordinatesY,int *gpu_meshNumber,int *gpu_translationCommand,int *gpu_amount,int *gpu_order,int *gpu_index,int *gpu_size_indexing){
  int operations[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
  //int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idx= blockIdx.x;
  if(idx<numTranslations){
    int ind = gpu_index[gpu_meshNumber[idx]];
    int op = gpu_translationCommand[idx];
    int times = gpu_amount[idx];

    int size = gpu_size_indexing[gpu_order[ind]];
    

    for(int i=ind;i<ind+size;i++){
      int x = operations[op][0]*times;
      int y = operations[op][1]*times;

      atomicAdd(&gpu_hGlobalCoordinatesX[gpu_order[i]],x);
      atomicAdd(&gpu_hGlobalCoordinatesY[gpu_order[i]],y);
    }
    
  }
}
//(hOffset,hCsr,order,0,index,1,size_indexing,visited)
int ordering(int *offset,int *csr,std::vector<int>&order,int u,std::vector<int>&index,int size,std::vector<int>&size_indexing,std::vector<int>&visited ){
    order.push_back(u);
    visited[u]=1;
    index[u] =order.size()-1;
   // int p = order.size()-1;
    // ++*id;
    for(int i=offset[u];i<offset[u+1];i++){
      if(!visited[csr[i]])
        size+=ordering(offset,csr,order,csr[i],index,1,size_indexing,visited);
    }
    size_indexing[u]=size;
    return size;
}


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

	int *gpu_hOffset,*gpu_hCsr, *gpu_hOpacity, **gpu_hMesh, *gpu_hGlobalCoordinatesX, *gpu_hGlobalCoordinatesY, *gpu_hFrameSizeX, *gpu_hFrameSizeY;
    cudaMalloc(&gpu_hOffset,sizeof(int)*(V+1));
    cudaMemcpy(gpu_hOffset,hOffset,sizeof(int)*(V+1),cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_hCsr,sizeof(int)*E);
    cudaMemcpy(gpu_hCsr,hCsr,sizeof(int)*E,cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_hOpacity,sizeof(int)*V);
    cudaMemcpy(gpu_hOpacity,hOpacity,sizeof(int)*V,cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_hMesh,sizeof(int *)*V);
    int **dummy = (int**)malloc(sizeof(int*)*V);
    for(int i=0;i<V;i++){
        int *temp;
        int size = hFrameSizeX[i]*hFrameSizeY[i];
        cudaMalloc(&temp,sizeof(int)*size);
        cudaMemcpy(temp,hMesh[i],sizeof(int)*size,cudaMemcpyHostToDevice);
        dummy[i] = temp;
    }
    cudaMemcpy(gpu_hMesh,dummy,sizeof(int*)*V,cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_hGlobalCoordinatesX,sizeof(int)*V);
    cudaMemcpy(gpu_hGlobalCoordinatesX,hGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_hGlobalCoordinatesY,sizeof(int)*V);
    cudaMemcpy(gpu_hGlobalCoordinatesY,hGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_hFrameSizeX,sizeof(int)*V);
    cudaMemcpy(gpu_hFrameSizeX,hFrameSizeX,sizeof(int)*V,cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_hFrameSizeY,sizeof(int)*V);
    cudaMemcpy(gpu_hFrameSizeY,hFrameSizeY,sizeof(int)*V,cudaMemcpyHostToDevice);
	
	
	//calculating indexing size and mapping
	  std::vector<int>order;//preorder
    int id = 0;
    std::vector<int>index(V,0);//offset
    std::vector<int>size_indexing(V,0);//how many childs are there for one node
    std::vector<int>visited(V,0);
    ordering(hOffset,hCsr,order,0,index,1,size_indexing,visited);

    // for(int i=0;i<V;i++){
    //   std::cout<<order[i]<<" ";
    // }
    // std::cout<<"\n";
    // for(int i=0;i<V;i++){
    //   std::cout<<index[i]<<" ";
    // }
    // std::cout<<"\n";
    // for(int i =0;i<V;i++){
    //   std::cout<<size_indexing[i]<<" - "<<i<<"\n";
    // }
    // std::cout<<"\n";
    //making 3 arrays to separating translations
    int *gpu_meshNumber, *gpu_translationCommand, *gpu_amount;
    cudaMalloc(&gpu_meshNumber,sizeof(int)*numTranslations);
    cudaMalloc(&gpu_translationCommand,sizeof(int)*numTranslations);
    cudaMalloc(&gpu_amount,sizeof(int)*numTranslations);
    
    int*temp_trans = (int* )malloc(sizeof(int)*numTranslations);
    int *temp_mesh = (int*)malloc(sizeof(int)*numTranslations);
    int *temp_am = (int*)malloc(sizeof(int)*numTranslations);
    for(int i=0;i<numTranslations;i++){
      temp_trans[i] = translations[i][1];
      temp_mesh[i] = translations[i][0];
      temp_am[i] = translations[i][2];
    }


    // for(int i=0;i<numTranslations;i++){
    //   cudaMemcpy(&gpu_meshNumber,&translations[i][0],sizeof(int),cudaMemcpyHostToDevice);
    //   cudaMemcpy(&gpu_translationCommand,&translations[i][1],sizeof(int),cudaMemcpyHostToDevice);
    //   cudaMemcpy(&gpu_amount,&translations[i][2],sizeof(int),cudaMemcpyHostToDevice);
    // }   

    cudaMemcpy(gpu_meshNumber,temp_mesh,sizeof(int)*numTranslations,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_translationCommand,temp_trans,sizeof(int)*numTranslations,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_amount,temp_am,sizeof(int)*numTranslations,cudaMemcpyHostToDevice);

    //allocating memory for vectors 
    int *gpu_order,*gpu_index,*gpu_size_indexing;
    cudaMalloc(&gpu_order,sizeof(int)*V);
    cudaMalloc(&gpu_index,sizeof(int)*V);
    cudaMalloc(&gpu_size_indexing,sizeof(int)*V);
    cudaMemcpy(gpu_order,order.data(),sizeof(int)*V,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_index,index.data(),sizeof(int)*V,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size_indexing,size_indexing.data(),sizeof(int)*V,cudaMemcpyHostToDevice);

    //kernel call for translations
    // int threadsPerBlock  = 1024;
    // int blocks = (numTranslations+threadsPerBlock-1)/threadsPerBlock;
    
    calculatingTranslations<<<numTranslations,1>>>(numTranslations,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY,gpu_meshNumber,gpu_translationCommand,gpu_amount,gpu_order,gpu_index,gpu_size_indexing);
    cudaDeviceSynchronize();
    //allocating memory to gpu_hFinalPng
    int *gpu_hFinalPng;
    cudaMalloc(&gpu_hFinalPng,sizeof(int)*frameSizeX*frameSizeY);
    cudaMemset(gpu_hFinalPng,-1,sizeof(int)*frameSizeX*frameSizeY);

    //kernel call for setting opacity
    int threads = 1024;
    int blocks = (V+threads-1)/threads;
    fixingOpacity<<<blocks,threads>>>(V,gpu_hFinalPng,frameSizeX,frameSizeY,gpu_hOpacity,gpu_hFrameSizeX,gpu_hFrameSizeY,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY);
    cudaDeviceSynchronize();
    //mapping the opacity and making hash table for opacity
    int *gpu_hashOpacity ;
    cudaMalloc(&gpu_hashOpacity,sizeof(int)*3*1e8);
    hashingOpacity<<<blocks,threads>>>(gpu_hOpacity,gpu_hashOpacity,V);
    cudaDeviceSynchronize();
    int blockPerGrid = ((frameSizeX*frameSizeY)+1023)/1024;
    calculatingFinalImage<<<blockPerGrid,1024>>>(gpu_hFinalPng,frameSizeX,frameSizeY,gpu_hashOpacity,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY,gpu_hFrameSizeY,gpu_hMesh);
    cudaMemcpy(hFinalPng,gpu_hFinalPng,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
