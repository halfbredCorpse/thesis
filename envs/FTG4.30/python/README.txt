March 6, 2018

After you have successfully installed Version 4.00 or later, here are two quick steps to start. If you need more information, please read the content right below "//-----."

1. Start FightingICE with argument “-—py4j”

2. Execute Main~.py
    e.g.) python Main_PyAIvsPyAI.py -n 3
    In this, case, you are able to do 3 games.

//——————————————————————————————————————————————————————————————————//

In FightingICE you can control the launching of games and the AIs in Python with PYJ4.
You just need to use these arguments to launch FightingICE.

--py4j --port PORT_NUMBER

The port is optional, and by default it is 4242.
Now FightingICE is expecting that you launch the python application. (Note that you can also directly launch the Java application from Python)

Here is the basic Python code that connects to the gateway server (on port 4242) and gets back the manager.

from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

The python AIs just use the same interface as the Java's one (AIInterface). And you can create a basic AI as follows:
------------
from py4j.java_gateway import get_field

class KickAI(object):
    def __init__(self, gateway):
        self.gateway = gateway
        
    def close(self):
        pass
        
    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
    	print(x)
    	print(y)
    	print(z)
    	
    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
    	pass
        
    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
            
        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
                
        return 0
        
    def input(self):
        # Return the input for the current frame
        return self.inputKey
        
    def processing(self):
        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
                self.isGameJustStarted = True
                return
                
        if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            
        self.inputKey.empty()
        self.cc.skillCancel()     

        # Just spam kick
        self.cc.commandCall("B")
                        
    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]

------------
Now that you have your AI, you have to register it to the manager as follows:
manager.registerAI("KickAI", KickAI(gateway))

And with that you can just start a new game or a series of games.
print("Start game")

game = manager.createGame("ZEN", "ZEN", "Machete", "KickAI", 3)
#Note that the last argument "3" is for specifying the number of games.
manager.runGame(game)

print("After game")
sys.stdout.flush()

print("End of games")
gateway.close_callback_server()
gateway.close()

The method runGame will just wait for the end of the game before returning. Note that you can't launch multiple games in parallel on the same Java application.

---
That's all folks!