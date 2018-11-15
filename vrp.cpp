#include<iostream>
#include<string.h>
#include<math.h>
#include <algorithm>

using namespace std;

typedef struct NODE{
    int node;
    // Instead of x and y co-ordinates costs can be given
    int x;
    int y;
    int d;
} Node;

typedef struct savings{
    int start;
    int end;
    int s_between;
} Savings;

typedef struct route{
    int nodes_in_route[1024];
    int nodesAdded;
} Route;

typedef struct Demand{
	int node;
	int d;
} Demand;

typedef struct keyVal{
	int key;
	int val;
	int routeIndex;
	int indexOfnodeInRouteInResultArray;
} keyVal;

void calculateSavings(int* costMatrix, int* hostSavingsMatrix, int rows, int cols){
    for(int x = 0; x < rows; x++){
        for(int y = 0; y < cols; y++){
            int valx = costMatrix[x];
            int valy = costMatrix[y];
            int valxy = costMatrix[x + y * rows];
            hostSavingsMatrix[x + y * rows] = (x != y && x > y) ? valx + valy - valxy : 0;
        }
    }
}

struct OBCmp {
    bool operator()(const Savings& o1, const Savings& o2) {
        return o1.s_between > o2.s_between;
    }
};

void sortSavings(Savings * obs, int N){
    sort(obs, obs+N, OBCmp());
}

void getCostMatrix(Node* nodeInfos, int *costMatrix, int rows, int cols){
    for(int x = 0; x < rows; x++){
        for(int y = 0; y < cols; y++){
            Node tempNode0 = nodeInfos[x];
            Node tempNode1 = nodeInfos[y];
            costMatrix[x*cols + y] = (x == y) ? 0 : 1 + int(sqrt((float)(((tempNode0.x-tempNode1.x)*(tempNode0.x - tempNode1.x))+((tempNode0.y - tempNode1.y)*(tempNode0.y - tempNode1.y)))));
        }
    }
}

int main(){

    int no_of_nodes;
    int vehicleCapacity;

    std::cout << "No of nodes " << endl;
    cin >> no_of_nodes;

    std::cout << "Capacity of each vehicle " << endl;
    cin >> vehicleCapacity;

    // Vector of (no_of_nodes + 1) * (no_of_nodes + 1) size
    int rows = (no_of_nodes + 1), cols = (no_of_nodes + 1);
    const int size = rows * cols;

    Node * hostN;

    int * hostCostMatrix;

    int * hostSavingsMatrix;

    Savings * hostSavingsMatrixRecord;

    hostCostMatrix = new int[size];

    // Assume we are starting from (0,0)
    hostN = new Node[no_of_nodes + 1];

    hostN[0].node = 0;
    hostN[0].x = 0;
    hostN[0].y = 0;
    hostN[0].d = 0;

    for(int i=0; i < no_of_nodes; i++){
        std::cout <<"Node Info for node (x, y, demand)" <<endl << "Node: " << i+1 << endl;
        hostN[i+1].node = i+1;
        cin >> hostN[i+1].x;
        cin >> hostN[i+1].y;
        cin >> hostN[i+1].d;
    }

    getCostMatrix(hostN, hostCostMatrix, no_of_nodes + 1, no_of_nodes + 1);

    hostSavingsMatrix = new int[size];

    calculateSavings(hostCostMatrix, hostSavingsMatrix, no_of_nodes + 1, no_of_nodes + 1);

    int count = 0;

    hostSavingsMatrixRecord = new Savings[rows * cols];

    for (int i = 1; i < rows-1; ++i){
		for (int j = i + 1; j < cols; ++j) {
            hostSavingsMatrixRecord[count].start = i;
            hostSavingsMatrixRecord[count].end = j;
            hostSavingsMatrixRecord[count].s_between = hostSavingsMatrix[i*cols + j];
            count++;
		}
    }

    sortSavings(hostSavingsMatrixRecord, count);

    for(int i=0; i < count; i++){
        std::cout << hostSavingsMatrixRecord[i].s_between << endl;
    }

    int nodeCount = no_of_nodes + 1, maxRouteCount = no_of_nodes;
    keyVal * hostResultDict =  new keyVal[nodeCount];

    // Which root for which nodeA
    for(int i=0; i<nodeCount; ++i){
		hostResultDict[i].key = hostN[i].node;
		hostResultDict[i].val = 0;
    }

    Route * hostRouteList = new Route[maxRouteCount];

    int nodesProcessed = 0;
    int routesAdded = 0;
    int totalSavings = 0;

    // For each savings
    for(int i = 0; i < count; i++){
        int start = hostSavingsMatrixRecord[i].start;
        int end = hostSavingsMatrixRecord[i].end;

        int demandStart = hostN[i].d;
        int demandEnd = hostN[i].d;

        if (demandStart + demandEnd <= vehicleCapacity){

            std::cout << nodesProcessed << endl;

            if(hostResultDict[start].val == 0 && hostResultDict[end].val == 0){
                hostRouteList[routesAdded].nodes_in_route[0]  = start;
                hostRouteList[routesAdded].nodes_in_route[1]  = end;
                hostRouteList[routesAdded].nodesAdded = 2;
                hostResultDict[start].val = 1;
                hostResultDict[end].val = 1;
                hostResultDict[start].routeIndex = routesAdded;
                hostResultDict[end].routeIndex = routesAdded;
                hostResultDict[start].indexOfnodeInRouteInResultArray = 0;
                hostResultDict[end].indexOfnodeInRouteInResultArray = 1;
                nodesProcessed += 2;
                routesAdded += 1;
            }
            else if(hostResultDict[start].val == 1 && hostResultDict[end].val == 0){
                int indexOfRoute = hostResultDict[start].routeIndex;
                int numberOfNodesInRoute = hostRouteList[indexOfRoute].nodesAdded;
                int total_demand = 0;
                total_demand += demandEnd;
                for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++){
                    total_demand += hostN[hostRouteList[indexOfRoute].nodes_in_route[temp_i]].d;
                }
                if (total_demand <= vehicleCapacity){
                    if (hostResultDict[start].indexOfnodeInRouteInResultArray == 0 || hostResultDict[start].indexOfnodeInRouteInResultArray == (hostRouteList[indexOfRoute].nodesAdded - 1)){
                        hostRouteList[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = end;
                        hostRouteList[indexOfRoute].nodesAdded += 1;
                        hostResultDict[end].val = 1;
                        hostResultDict[end].routeIndex = indexOfRoute;
                        hostResultDict[end].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
                        nodesProcessed += 1;
                    }
                }
            }
            else if (hostResultDict[start].val == 0 && hostResultDict[end].val == 1){
                int indexOfRoute = hostResultDict[end].routeIndex;
                int numberOfNodesInRoute = hostRouteList[indexOfRoute].nodesAdded;
                int total_demand = 0;
                total_demand += demandStart;
                for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++){
                    total_demand += hostN[hostRouteList[indexOfRoute].nodes_in_route[temp_i]].d;
                }
                if (total_demand <= vehicleCapacity){
                    if (hostResultDict[end].indexOfnodeInRouteInResultArray == 0 || hostResultDict[end].indexOfnodeInRouteInResultArray == (hostRouteList[indexOfRoute].nodesAdded - 1)){
                        hostRouteList[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = start;
                        hostRouteList[indexOfRoute].nodesAdded += 1;
                        hostResultDict[start].val = 1;
                        hostResultDict[start].routeIndex = indexOfRoute;
                        hostResultDict[start].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
                        nodesProcessed += 1;
                    }
                }
            }
            std::cout << start << end << endl;
            std::cout << hostResultDict[start].val << hostResultDict[end].val << endl;
        }
    }

    for (int j = 1; j < nodeCount; j++){
        if (hostResultDict[j].val == 0){
            hostRouteList[routesAdded].nodes_in_route[0] = hostN[j].node;
            hostRouteList[routesAdded].nodesAdded = 1;
            nodesProcessed += 1;
            routesAdded += 1;
        }
    }

    for (int i = 0; i < routesAdded; i++){

		Route temproute = hostRouteList[i];
		int localSavings = 0;
		int node1 = 0;
		int node2 = 0;
		int decisionMaker = 0;
		std::cout << "\nRoute\t\t:" << i << endl;
		std::cout <<"NodesAdded\t: "<<  temproute.nodesAdded <<endl << endl << "[\t";

        for (int j = 0; j < temproute.nodesAdded; j++){
            std::cout << temproute.nodes_in_route[j] << "\t" ;
            if (decisionMaker == 0){
                if (node1 != 0) {
                    node1 = temproute.nodes_in_route[j];
                    localSavings += *(hostSavingsMatrix + node2*cols + node1);
                }
                else{
                    node1 = temproute.nodes_in_route[j];
                }
                decisionMaker = 1;
            }
            else{
                node2 = temproute.nodes_in_route[j];
                decisionMaker = 0;
                localSavings += *(hostSavingsMatrix + node1*cols + node2);
            }
        }
        if (node2 == 0){
            localSavings = *(hostSavingsMatrix + node1);
        }
        std::cout << "]" << endl;
        decisionMaker = 0;
        totalSavings += localSavings;
        std::cout << "Savings: " << localSavings;
    }

    std::cout << "\nTotal Nodes Processed:" << nodesProcessed;
	std::cout << "\nTotal Savings:" << totalSavings << endl;

    return 0;
}

