from function import continueFunction as cF
import numpy as np
import time
import sys
import copy


class cmaes:

    def __init__(self):
        self.Name = "cmaes"
        self.FERuntime = 0
        self.FENum = 0
        self.setParameters()

    def setParameters(self):
        pass

    def optimize(self, cfp, ap, printLog=True):
        runtimeStart = time.clock()
        self.mainLoop(cfp, ap, printLog)
        self.runtime = time.clock() - runtimeStart

    def mainLoop(self, cfp, ap, printLog):
        print(ap.FEMax)
        np.random.seed(ap.initialSeed)
        Dim = cfp.funcDim
        function = getattr(cF, cfp.funcName)
        
        # object parameter start point
        xmeanw = (cfp.funcInitUpperBound  - cfp.funcInitLowerBound) * \
            np.random.random_sample((Dim, )) + cfp.funcInitLowerBound
        
        # initial step size, minimal step size
        sigma = 0.25;
        minsigam = 1e-15
        maxsigma = max(cfp.funcUpperBound - cfp.funcLowerBound) / np.sqrt(Dim)
        # initial phase
        flagInitPhase = 1

        # parameter setting selection
        lamb = (4 + np.floor(3 * np.log(Dim))).astype(int)
        mu = np.floor(lamb / 2).astype(int)
        # muXone array for weighted recomb
        arweights = np.log((lamb + 1)/2) - np.log(np.array(range(1, mu+1)))
        # parameter setting: adaptation
        cc = 4 / (Dim + 4)
        ccov = 2 / (Dim + 2**0.5)**2
        cs = 4 / (Dim + 4)
        damp = (1 - min(0.7, Dim * lamb/ap.FEMax)) / cs + 1

        # initialize dynamic strategy parameters and constants
        B = np.eye(Dim)
        D = np.eye(Dim)
        BD = np.dot(B, D)
        C = np.dot(BD, np.transpose(BD))
        pc = np.zeros((Dim, 1))
        ps = np.zeros((Dim, 1))
        cw = np.sum(arweights) / np.linalg.norm(arweights)
        chiN = Dim**0.5 * (1 - 1/(Dim * 4) + 1/(21 * Dim**2) )

        self.FENum = 0
        
        lowerBoundX = np.kron(np.ones((lamb, 1)), cfp.funcLowerBound)
        lowerBoundX = np.transpose(lowerBoundX)
        upperBoundX = np.kron(np.ones((lamb, 1)), cfp.funcUpperBound)
        upperBoundX = np.transpose(upperBoundX)

        k = 1
        while self.FENum < ap.FEMax:
            
            # generate and evaluate lambda offspring
            arz = np.random.randn(Dim, lamb)
            arx = np.dot(xmeanw[:, np.newaxis], np.ones((1, lamb))) + sigma * (np.dot(BD, arz))


            # Handle the elements of the variable which violate the boundary
            position = arx > upperBoundX            
            arx[position] = 2 * upperBoundX[position] - arx[position]
            # position_aa = arx[position] < lowerBoundX[position]
            # arx[position[position_aa]] = lowerBoundX(position[position_aa])

            position = arx < lowerBoundX
            arx[position] = 2 * lowerBoundX[position] - arx[position]
            # position_aa = arx[position] > upperBoundX[position]
            # arx[position[position_aa]] = upperBoundX(position[position_aa])
            
            start = time.clock()
            arfitness = function(arx.T)
            self.FERuntime += (time.clock() - start)
            self.FENum += lamb

            arindex = np.argsort(arfitness)
            arfitness.sort()
            xold = copy.deepcopy(xmeanw)
            xmeanw = np.dot(arx[:, arindex[0:mu]], arweights) / sum(arweights)
            zmeanw = np.dot(arz[:, arindex[0:mu]], arweights) / sum(arweights)

            # Adapt covariance matrix
            pc = (1-cc) * pc + (np.sqrt(cc * (2-cc)) * cw / sigma) * (xmeanw[:, np.newaxis] - xold[:, np.newaxis])
            if not flagInitPhase: # do not adapt in the initial phase
                C = (1 - ccov) * C + ccov * np.dot(pc, np.transpose(pc))
            
            #adaot sigma
            tmp = (np.sqrt(cs * (2-cs)) * cw) * (np.dot(B, zmeanw))
            ps = (1-cs) * ps + tmp[:, np.newaxis]
            sigma = sigma * np.exp((np.linalg.norm(ps) - chiN) / chiN / damp)

            # update B and D and C
            if (self.FENum/lamb) % (1/ccov/Dim/5) < 1:
                # enforce symmetry
                C = np.triu(C) + np.transpose(np.triu(C, 1))
                [D, B] = np.linalg.eig(C)
                # limit condition of C to 1e14 + 1
                if max(D) > (1e14 * min(D)):
                    tmp = max(D) / 1e14 - min(D)
                    C = C + tmp*np.eye(Dim)
                    D = D + tmp
                # D contains standard deviations now
                D = np.diag(np.sqrt(D))
                # for speed up only
                BD = np.dot(B, D)
            
            # adjust minimal size
            if sigma * min(np.diag(D)) < minsigam or arfitness[0] == arfitness[min(mu+1, lamb)] or (xmeanw == xmeanw + 0.2 * sigma * BD[:, np.floor(((self.FENum)/lamb)%Dim).astype(int)]).all():
                sigma = 1.4 * sigma
            
            if sigma > maxsigma:
                sigma = maxsigma
            
            # test for end of initial phase
            if flagInitPhase and (self.FENum/lamb) > (2/cs):
                if ( np.linalg.norm(ps) - chiN ) / chiN < 0.05:
                    flagInitPhase = 0
            # print(arfitness[0])
        self.optimalX = arx[:, arindex[0]]
        self.optimalY = arfitness[0]
        if printLog:
            # summary
            print('$--------Result--------$\n')
            print('*Function: {0}\tDimension: {1}\t FEMax: {2}\n'.format(
                cfp.funcName, cfp.funcDim, self.FENum))
            print('Optimal Y  : {0} \n'.format(self.optimalY))  
            
            


                



            




        