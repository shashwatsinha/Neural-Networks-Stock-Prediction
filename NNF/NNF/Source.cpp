#define _CRT_SECURE_NO_DEPRECATE
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Definitions.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <list>

using namespace std;



typedef struct {          // A LAYER OF A NET:                     
	int    Units;         // number of units in this layer       
	double*  Output;       // output of ith unit                  
	double*  Error;          // error term of ith unit              
	double** Weight;        // connection weights to ith unit      
	double** WeightSave;    // saved weights for stopped training  
	double** dWeight;       // last weight deltas for error   
} LAYER;

typedef struct {                 // A NET:                            
	LAYER**       Layer;         // layers of this net                  
	LAYER*        InputLayer;    // input layer                         
	LAYER*        OutputLayer;   // output layer                        
	double          Alpha;        // error factor                     
	double          Eta;           // learning rate                       
	double          Gain;        // gain of sigmoid function            
	double          Error;      // total net error                     
} NET;


//Randoms from Distributions
void InitializeRandoms()
{
	srand(4711);
}

int RandomEqualINT(int Low, int High)
{
	return rand() % (High - Low + 1) + Low;
}

double RandomEqualREAL(double Low, double High)
{
	return ((double)rand() / RAND_MAX) * (High - Low) + Low;
}

//Code for the Application


int  Units[NUM_LAYERS] = { 50, 40,30,20, M };



double  Values_[NUM_YEARS];
double  Values[NUM_YEARS];

double  Mean;
double  TrainError;
double Min, Max;
double  TrainErrorPredictingMean;
double  TestError;
double  TestErrorPredictingMean;
double a[NUM_YEARS];


FILE*   f;

void NormalizeValues()
{
	int  Year;
	//double Min, Max;

	Min = MAX_REAL;
	Max = MIN_REAL;
	for (Year = 0; Year<NUM_YEARS; Year++) {
		Min = MIN(Min, Values[Year]);
		Max = MAX(Max, Values[Year]);
	}
	cout << Min <<" "<< Max;
	Mean = 0;
	for (Year = 0; Year<NUM_YEARS; Year++) {
		Values_[Year] =  Values[Year] =((Values[Year] - Min) / (Max - Min)) * (HI - LO) + LO;
		Mean += Values[Year] / NUM_YEARS;
	}
}

void InitializeApplication(NET* Net)
{
	int  Year, i;
	double Out, Err;

	Net->Alpha = 0.5;
	Net->Eta = 0.05;
	Net->Gain = 1;

	NormalizeValues();
	TrainErrorPredictingMean = 0;
	for (Year = TRAIN_LWB; Year <= TRAIN_UPB; Year++) {
		for (i = 0; i<M; i++) {
			Out = Values[Year + i];
			Err = Mean - Out;
			TrainErrorPredictingMean += 0.5 * sqr(Err);
		}
	}
	TestErrorPredictingMean = 0;
	for (Year = TEST_LWB; Year <= TEST_UPB; Year++) {
		for (i = 0; i < M; i++) {
			Out = Values[Year + i];
			Err = Mean - Out;
			TestErrorPredictingMean += 0.5 * sqr(Err);
		}
	}
	f = fopen("BPN.txt", "w");
}

void FinalizeApplication(NET* Net)
{
	fclose(f);
}

//Initialize the values 

void GenerateNetwork(NET* Net)
{
	int l, i;

	Net->Layer = (LAYER**)calloc(NUM_LAYERS, sizeof(LAYER*));

	for (l = 0; l<NUM_LAYERS; l++) {
		Net->Layer[l] = (LAYER*)malloc(sizeof(LAYER));

		Net->Layer[l]->Units = Units[l];
		Net->Layer[l]->Output = (double*)calloc(Units[l] + 1, sizeof(double));
		Net->Layer[l]->Error = (double*)calloc(Units[l] + 1, sizeof(double));
		Net->Layer[l]->Weight = (double**)calloc(Units[l] + 1, sizeof(double*));
		Net->Layer[l]->WeightSave = (double**)calloc(Units[l] + 1, sizeof(double*));
		Net->Layer[l]->dWeight = (double**)calloc(Units[l] + 1, sizeof(double*));
		Net->Layer[l]->Output[0] = 1;

		if (l != 0) {
			for (i = 1; i <= Units[l]; i++) {
				Net->Layer[l]->Weight[i] = (double*)calloc(Units[l - 1] + 1,
					sizeof(double));
				Net->Layer[l]->WeightSave[i] = (double*)calloc(Units[l - 1] + 1,
					sizeof(double));
				Net->Layer[l]->dWeight[i] = (double*)calloc(Units[l - 1] + 1,
					sizeof(double));
			}
		}
	}
	Net->InputLayer = Net->Layer[0];
	Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
	Net->Alpha = 0.9;
	Net->Eta = 0.25;
	Net->Gain = 1;
}

void RandomWeights(NET* Net)
{
	int l, i, j;

	for (l = 1; l<NUM_LAYERS; l++) {
		for (i = 1; i <= Net->Layer[l]->Units; i++) {
			for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
				Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
			}
		}
	}
}

void SetInput(NET* Net, double* Input)
{
	int i;

	for (i = 1; i <= Net->InputLayer->Units; i++) {
		Net->InputLayer->Output[i] = Input[i - 1];
	}
}

void GetOutput(NET* Net, double* Output)
{
	int i;

	for (i = 1; i <= Net->OutputLayer->Units; i++) {
		Output[i - 1] = Net->OutputLayer->Output[i];
	}
}

//Support for the stopped training

void SaveWeights(NET* Net)
{
	int l, i, j;

	for (l = 1; l<NUM_LAYERS; l++) {
		for (i = 1; i <= Net->Layer[l]->Units; i++) {
			for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
				Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
			}
		}
	}
}

void RestoreWeights(NET* Net)
{
	int l, i, j;

	for (l = 1; l<NUM_LAYERS; l++) {
		for (i = 1; i <= Net->Layer[l]->Units; i++) {
			for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
				Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
			}
		}
	}
}

//Propogation Signals

void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
	int  i, j;
	double Sum;

	for (i = 1; i <= Upper->Units; i++) {
		Sum = 0;
		for (j = 0; j <= Lower->Units; j++) {
			Sum += Upper->Weight[i][j] * Lower->Output[j];
		}
		Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
	}
}

void PropagateNet(NET* Net)
{
	int l;

	for (l = 0; l<NUM_LAYERS - 1; l++) {
		PropagateLayer(Net, Net->Layer[l], Net->Layer[l + 1]);
	}
}

//BackPropogation algo

void ComputeOutputError(NET* Net, double* Target)
{
	int  i;
	double Out, Err;

	Net->Error = 0;
	for (i = 1; i <= Net->OutputLayer->Units; i++) {
		Out = Net->OutputLayer->Output[i];
		Err = Target[i - 1] - Out;
		Net->OutputLayer->Error[i] = Net->Gain * Out * (1 - Out) * Err;
		Net->Error += 0.5 * sqr(Err);
	}
}

void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
	int  i, j;
	double Out, Err;

	for (i = 1; i <= Lower->Units; i++) {
		Out = Lower->Output[i];
		Err = 0;
		for (j = 1; j <= Upper->Units; j++) {
			Err += Upper->Weight[j][i] * Upper->Error[j];
		}
		Lower->Error[i] = Net->Gain * Out * (1 - Out) * Err;
	}
}

void BackpropagateNet(NET* Net)
{
	int l;

	for (l = NUM_LAYERS - 1; l>1; l--) {
		BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l - 1]);
	}
}

void AdjustWeights(NET* Net)
{
	int  l, i, j;
	double Out, Err, dWeight;

	for (l = 1; l<NUM_LAYERS; l++) {
		for (i = 1; i <= Net->Layer[l]->Units; i++) {
			for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
				Out = Net->Layer[l - 1]->Output[j];
				Err = Net->Layer[l]->Error[i];
				dWeight = Net->Layer[l]->dWeight[i][j];
				Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha *
					dWeight;
				Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
			}
		}
	}
}

//Simulating Net

void SimulateNet(NET* Net, double* Input, double* Output, double* Target, int
	Training)
{
	SetInput(Net, Input);
	PropagateNet(Net);
	GetOutput(Net, Output);

	ComputeOutputError(Net, Target);
	if (Training) {
		BackpropagateNet(Net);
		AdjustWeights(Net);
	}
}

void TrainNet(NET* Net, int a)
{
	int  Year, n;
	double Output[M];

	for (n = 0; n<a*TRAIN_YEARS; n++) {
		Year = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);
		SimulateNet(Net, &(Values[Year - N]), Output, &(Values[Year]), TRUE);
	}
}

void TestNet(NET* Net)
{
	int Year;
	double Output[M];

	TrainError = 0;
	for (Year = TRAIN_LWB; Year <= TRAIN_UPB; Year++) {
		SimulateNet(Net, &(Values[Year - N]), Output, &(Values[Year]), FALSE);
		TrainError += Net->Error;
	}
	TestError = 0;
	for (Year = TEST_LWB; Year <= TEST_UPB; Year++) {
		SimulateNet(Net, &(Values[Year - N]), Output, &(Values[Year]), FALSE);
		TestError += Net->Error;
	}
	
}

void EvaluateNet(NET* Net)
{
	int  Year;
	double Output[M];
	double Output_[M];

	fprintf(f, "\n\n\n");
	fprintf(f, "Year     Original Values   Predicted Values    \n");
	fprintf(f, "\n");
	for (Year = EVAL_LWB; Year <= EVAL_UPB; Year++) {
		SimulateNet(Net, &(Values[Year - N]), Output, &(Values[Year]), FALSE);
		SimulateNet(Net, &(Values_[Year - N]), Output_, &(Values_[Year]), FALSE);
		Values_[Year] = Output_[0];
		cout << Values_[Year];
		fprintf(f, "%d            %0.02f          %0.002f                 %f         \n",
			FIRST_YEAR + Year,
			((Values[Year] - LO) *((Max - Min) / (HI - LO)) + Min),
			((Values_[Year] - LO) *((Max - Min) / (HI - LO)) + Min),
			(((Values[Year] - LO) *((Max - Min) / (HI - LO)) + Min) - ((Values_[Year] - LO) *((Max - Min) / (HI - LO)) + Min)));
			
	}
}


void split_line(string& line, string delim, list<string>& values)
{
	size_t pos = 0;
	while ((pos = line.find(delim, (pos + 1))) != string::npos) {
		string p = line.substr(0, pos);
		values.push_back(p);
		line = line.substr(pos + 1);
	}

	if (!line.empty()) {
		values.push_back(line);
	}
}


//The main function

void main()
{
	NET  Net;
	int Stop;
	double MinTestError;

	
	ifstream in;
	
	in.open("fileoutput.txt");

	for (int i = 0; i < NUM_YEARS; i++) {
		in >> Values[i] ;
	
	}

	InitializeRandoms();
	GenerateNetwork(&Net);
	RandomWeights(&Net);
	InitializeApplication(&Net);

	Stop = FALSE;
	MinTestError = MAX_REAL;
	do {
		TrainNet(&Net, 10);
		TestNet(&Net);
		if (TestError < MinTestError) {
			
			MinTestError = TestError;
			SaveWeights(&Net);
		}
		else if (TestError > 1.2 * MinTestError) {
			
			Stop = TRUE;
			RestoreWeights(&Net);
		}
	} while (NOT Stop);

	TestNet(&Net);
	EvaluateNet(&Net);

	FinalizeApplication(&Net);
}