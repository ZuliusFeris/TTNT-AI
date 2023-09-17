using System;
using QLearningFramework;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace QLearningFramework
{
    /// <summary>
    /// Version 0.1
    /// Author: Kunuk Nykjaer
    /// one to many relationships data structure
    /// QLearning [1 -- *] State [1 -- *] Action [1 -- *] ActionResult
    /// </summary>
    class QLearning // ds trạng thái(state), giới hạn số lần học (MaxQ)
    {
        #region khai báo các biến qlearning
        public List<QState> States { get; private set; } //danh sách các trạng thái
        public Dictionary<string, QState> StateLookup { get; private set; }// từ điển truy cập các trạng thái

        public double Alpha { get; internal set; } // tỷ lệ học
        public double Gamma { get; internal set; } // tỷ lệ chính sát

        public HashSet<string> EndStates { get; private set; } // trạng thái kết thúc
        public int MaxExploreStepsWithinOneEpisode { get; internal set; } //giới hạn số lần học
        public bool ShowWarning { get; internal set; } // show runtime warnings regarding q-learning
        public int Episodes { get; internal set; } // số lượng tập dữ liệu sau quá trình học
        #endregion
        public QLearning()
        {
            States = new List<QState>();
            StateLookup = new Dictionary<string, QState>();
            EndStates = new HashSet<string>();

            // Default when not set
            MaxExploreStepsWithinOneEpisode = 1000;
            Episodes = 1000;
            Alpha = 0.1;
            Gamma = 0.9;
            ShowWarning = true;
        }

        public void AddState(QState state) // hàm thêm trạng thái
        {
            States.Add(state);
        }

        public void RunTraining()
        {
            QMethod.Validate(this);

            /*       
            For each episode: Select random initial state 
            Do while not reach goal state
                Select one among all possible actions for the current state 
                Using this possible action, consider to go to the next state 
                Get maximum Q value of this next state based on all possible actions                
                Set the next state as the current state
            */

            // For each episode
            var rand = new Random();
            long maxloopEventCount = 0;

            // Train episodes
            for (long i = 0; i < Episodes; i++)
            {
                long maxloop = 0;
                // Select random initial state          
                int stateIndex = rand.Next(States.Count);
                QState state = States[stateIndex];
                QAction action = null;
                do
                {
                    if (++maxloop > MaxExploreStepsWithinOneEpisode)
                    {
                        if (ShowWarning)
                        {
                            string msg = string.Format(
                            "{0} !! MAXLOOP state: {1} action: {2}, {3} endstate is to difficult to reach?",
                            ++maxloopEventCount, state, action, "maybe your path setup is wrong or the ");
                            QMethod.Log(msg);
                        }

                        break;
                    }

                    // no actions, skip this state
                    if (state.Actions.Count == 0)
                        break;

                    // Selection strategy is random based on probability
                    int index = rand.Next(state.Actions.Count);
                    action = state.Actions[index];

                    // Using this possible action, consider to go to the next state
                    // Pick random Action outcome
                    QActionResult nextStateResult = action.PickActionByProbability();
                    string nextStateName = nextStateResult.StateName;

                    double q = nextStateResult.QEstimated;
                    double r = nextStateResult.Reward;
                    double maxQ = MaxQ(nextStateName);

                    // Q(s,a)= Q(s,a) + alpha * (R(s,a) + gamma * Max(next state, all actions) - Q(s,a))
                    double value = q + Alpha * (r + Gamma * maxQ - q); // q-learning                  
                    nextStateResult.QValue = value; // update

                    // is end state go to next episode
                    if (EndStates.Contains(nextStateResult.StateName))
                        break;

                    // Set the next state as the current state                    
                    state = StateLookup[nextStateResult.StateName];

                } while (true);
            }
        }


        double MaxQ(string stateName)
        {
            const double defaultValue = 0;

            if (!StateLookup.ContainsKey(stateName))
                return defaultValue;

            QState state = StateLookup[stateName];
            var actionsFromState = state.Actions;
            double? maxValue = null;
            foreach (var nextState in actionsFromState)
            {
                foreach (var actionResult in nextState.ActionsResult)
                {
                    double value = actionResult.QEstimated;
                    if (value > maxValue || !maxValue.HasValue)
                        maxValue = value;
                }
            }

            // no update
            if (!maxValue.HasValue && ShowWarning)
                QMethod.Log(string.Format("Warning: No MaxQ value for stateName {0}",
                    stateName));

            return maxValue.HasValue ? maxValue.Value : defaultValue;
        }

        public void PrintQLearningStructure()
        {
            Console.WriteLine("** Q-Learning structure **");
            foreach (QState state in States)
            {
                Console.WriteLine("State {0}", state.StateName);
                for (int i = 0; i < state.Actions.Count; i++)
                {
                    for (int j = i + 1; j < state.Actions.Count; j++)
                    {
                        if (state.Actions[i].ActionsResult[0].QValue < state.Actions[j].ActionsResult[0].QValue)
                        {
                            var a = state.Actions[j];
                            state.Actions[j] = state.Actions[i];
                            state.Actions[i] = a;
                        }
                    }
                }

                
                foreach (QAction action in state.Actions)
                {
                    Console.WriteLine("  Action " + action.ActionName);
                    Console.Write(action.GetActionResults());
                }
            }
            Console.WriteLine();
        }

        public void ShowPolicy()
        {
            Console.WriteLine("** Show Policy **");
            foreach (QState state in States)
            {
                double max = Double.MinValue;
                string actionName = "nothing";
                foreach (QAction action in state.Actions)
                {
                    foreach (QActionResult actionResult in action.ActionsResult)
                    {
                        if (actionResult.QEstimated > max)
                        {
                            max = actionResult.QEstimated;
                            actionName = action.ActionName.ToString();
                        }
                    }
                }

                Console.WriteLine(string.Format("From state {0} do action {1}, max QEstimated is {2} \n",
                    state.StateName, actionName, max.Pretty()));
            }
        }
        

    }

    class QState // gồm có tên và hành động {stateName; Action}
    {
        public string StateName { get; private set; } // tên trạng thái
        public List<QAction> Actions { get; private set; }// danh sách các hành động

        public void AddAction(QAction action)
        {
            Actions.Add(action);
        }

        public QState(string stateName, QLearning q)
        {
            q.StateLookup.Add(stateName, this);
            StateName = stateName;
            Actions = new List<QAction>();
        }

        public override string ToString()
        {
            return string.Format("StateName {0}", StateName);
        }
    }

    class QAction
    {
        private static readonly Random Rand = new Random();
        public QActionName ActionName { get; internal set; } // tên action
        public string CurrentState { get; private set; } // trạng thái hiện tại
        public List<QActionResult> ActionsResult { get; private set; } //kết quả của hành động

        public void AddActionResult(QActionResult actionResult)
        {
            ActionsResult.Add(actionResult);
        }

        public string GetActionResults()
        {
            var sb = new StringBuilder();
            foreach (QActionResult actionResult in ActionsResult)
                sb.AppendLine("     ActionResult " + actionResult);

            return sb.ToString();
        }

        public QAction(string currentState, QActionName actionName = null)
        {
            CurrentState = currentState;
            ActionsResult = new List<QActionResult>();
            ActionName = actionName;
        }

        // The sum of action outcomes must be close to 1
        public void ValidateActionsResultProbability()
        {
            const double epsilon = 0.1;

            if (ActionsResult.Count == 0)
                throw new ApplicationException(string.Format(
                    "ValidateActionsResultProbability is invalid, no action results:\n {0}",
                    this));

            double sum = ActionsResult.Sum(a => a.Probability);
            if (Math.Abs(1 - sum) > epsilon)
                throw new ApplicationException(string.Format(
                    "ValidateActionsResultProbability is invalid:\n {0}", this));
        }

        public QActionResult PickActionByProbability()
        {
            double d = Rand.NextDouble();
            double sum = 0;
            foreach (QActionResult actionResult in ActionsResult)
            {
                sum += actionResult.Probability;
                if (d <= sum)
                    return actionResult;
            }

            // we might get here if sum probability is below 1.0 e.g. 0.99 
            // and the d random value is 0.999
            if (ActionsResult.Count > 0)
                return ActionsResult.Last();

            throw new ApplicationException(string.Format("No PickAction result: {0}", this));
        }

        public override string ToString()
        {
            double sum = ActionsResult.Sum(a => a.Probability);
            return string.Format("ActionName {0} probability sum: {1} actionResultCount {2}",
                ActionName, sum, ActionsResult.Count);
        }
    }

    class QActionResult
    {
        public string StateName { get; internal set; }
        public string PrevStateName { get; internal set; }
        public double QValue { get; internal set; } // Q value is stored here        
        public double Probability { get; internal set; }
        public double Reward { get; internal set; }

        public double QEstimated
        {
            get { return QValue * Probability; }
        }

        public QActionResult(string CurrentState, string stateNameNext = null,
            double probability = 1, double reward = 0)
        {
            PrevStateName = CurrentState;
            StateName = stateNameNext;
            Probability = probability;
            Reward = reward;
        }

        public override string ToString()
        {
            return string.Format("State {0}, Prob. {1}, Reward {2}, PrevState {3}, QE {4}",
                StateName, Probability.Pretty(), Reward, PrevStateName, QEstimated.Pretty());
        }
    }

    public class QActionName
    {
        public string From { get; private set; }
        public string To { get; private set; }

        public QActionName(string from, string to = null)
        {
            From = from;
            To = to;
        }

        public override string ToString()
        {
            return GetActionName();
        }

        public string GetActionName()
        {
            if (To == null)
                return From;
            return QMethod.ActionNameFromTo(From, To);
        }
    }

    static class QMethod
    {
        public static void Log(string s)
        {
            Console.WriteLine(s);
        }

        public static readonly CultureInfo CultureEnUs = new CultureInfo("en-US");

        public static string ToStringEnUs(this double d)
        {
            return d.ToString("G", CultureEnUs);
        }
        public static string Pretty(this double d)
        {
            return ToStringEnUs(Math.Round(d, 2));
        }

        public static string ActionNameFromTo(string a, string b)
        {
            return string.Format("from_{0}_to_{1}", a, b);
        }

        public static string EnumToString<T>(this T type)
        {
            return Enum.GetName(typeof(T), type);
        }

        public static void ValidateRange(double d, string origin = null)
        {
            if (d < 0 || d > 1)
            {
                string s = origin ?? string.Empty;
                throw new ApplicationException(string.Format("ValidateRange error: {0} {1}", d, s));
            }
        }

        public static void Validate(QLearning q)
        {
            foreach (var state in q.States)
            {
                foreach (var action in state.Actions)
                {
                    action.ValidateActionsResultProbability();
                }
            }
        }
    }
}

namespace ConsoleQLearning
{
    class PathFindingDemo
    {
        // ----------- Insert the state names here -----------
        internal enum StateNameEnum
        {
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U,
            V, W, X, Y, Z
        }
        // ----------- End Insert the state names here -------

        static void Main(string[] args)
        {
            DateTime starttime = DateTime.Now;
            Console.WriteLine("A   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n");

            StateNameEnum sne;
            string r;
            do
            {
                Console.Write("Ponit END: ");
                r = Console.ReadLine();
                if (Enum.TryParse(r.ToUpper(), out sne))
                    PathFinding(r);
                else
                    break;

            } while (true);

            

            double timespend = DateTime.Now.Subtract(starttime).TotalSeconds;
            Console.WriteLine("\n{0} sec. press a key ...", timespend.Pretty()); Console.ReadKey();
        }

        static void PathFinding(string etrs)
        {
            QLearning q = new QLearning
            {
                Episodes = 1000,
                Alpha = 0.1,
                Gamma = 0.9,
                MaxExploreStepsWithinOneEpisode = 1000
            };

            QAction fromTo;
            QState state;
            string stateName;
            string stateNameNext;

            // ----------- Begin Insert the path setup here -----------
            /*
A	B	C	D
E	F	G	H
I	J	K	L
M	N	O	P
			
0	-1	100	0
0	-1	-1	0
0	0	0	0
0	0	0	0

             * */
            switch (etrs.ToUpper()) {
                case "A":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "100 - 1   0  0 \n" +
                            "0   - 1 - 1  0 \n" +
                            "0     0   0  0 \n" +
                            "0     0   0  0 \n");
                    #region end A (BGF=wall)
                    q.EndStates.Add(StateNameEnum.A.EnumToString());

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "C":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   100  0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    0 \n" +
                            "0     0   0    0 \n");
                    #region end C (BGF=wall)
                    q.EndStates.Add(StateNameEnum.C.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "D":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0  100 \n" +
                            "0   - 1 - 1  0 \n" +
                            "0     0   0  0 \n" +
                            "0     0   0  0 \n");
                    #region end D (BGF=wall)
                    q.EndStates.Add(StateNameEnum.D.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "E":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0  0 \n" +
                            "100 - 1 - 1  0 \n" +
                            "0     0   0  0 \n" +
                            "0     0   0  0 \n");
                    #region end E (BGF=wall)
                    q.EndStates.Add(StateNameEnum.E.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "H":
                    Console.WriteLine("\nA   B   C   D \n" +
                           "E   F   G   H \n" +
                           "I   J   K   L \n" +
                           "M   N   O   P \n" +
                           "\n\n\n" +
                           "0   - 1   0    0 \n" +
                           "0   - 1 - 1    100 \n" +
                           "0     0   0    0 \n" +
                           "0     0   0    0 \n");
                    #region end H (BGF=wall)
                    q.EndStates.Add(StateNameEnum.H.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    
                    /*
                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "I":
                    Console.WriteLine("\nA   B   C   D \n" +
                           "E   F   G   H \n" +
                           "I   J   K   L \n" +
                           "M   N   O   P \n" +
                           "\n\n\n" +
                           "0   - 1   0    0 \n" +
                           "0   - 1 - 1    0 \n" +
                           "100   0   0    0 \n" +
                           "0     0   0    0 \n");
                    #region end I (BGF=wall)
                    q.EndStates.Add(StateNameEnum.I.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "J":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0   100   0    0 \n" +
                            "0     0   0    0 \n");
                    #region end J (BGF=wall)
                    q.EndStates.Add(StateNameEnum.J.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "K":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   100  0 \n" +
                            "0     0   0    0 \n");
                    #region end K (BGF=wall)
                    q.EndStates.Add(StateNameEnum.K.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 10 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "L":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    100 \n" +
                            "0     0   0    0 \n");
                    #region end L (BGF=wall)
                    q.EndStates.Add(StateNameEnum.L.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    /*
                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "M":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    0 \n" +
                            "100   0   0    0 \n");
                    #region end M (BGF=wall)
                    q.EndStates.Add(StateNameEnum.M.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "N":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    0 \n" +
                            "0   100   0    0 \n");
                    #region end N (BGF=wall)
                    q.EndStates.Add(StateNameEnum.N.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "O":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    0 \n" +
                            "0     0 100    0 \n");
                    #region end O (BGF=wall)
                    q.EndStates.Add(StateNameEnum.O.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
                case "P":
                    Console.WriteLine("\nA   B   C   D \n" +
                            "E   F   G   H \n" +
                            "I   J   K   L \n" +
                            "M   N   O   P \n" +
                            "\n\n\n" +
                            "0   - 1   0    0 \n" +
                            "0   - 1 - 1    0 \n" +
                            "0     0   0    0 \n" +
                            "0     0   0    100 \n");
                    #region end P (BGF=wall)
                    q.EndStates.Add(StateNameEnum.P.EnumToString());

                    // State A           
                    stateName = StateNameEnum.A.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action A -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State D           
                    stateName = StateNameEnum.D.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action D -> C
                    stateNameNext = StateNameEnum.C.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action D -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    
                    // State C           
                    stateName = StateNameEnum.C.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action C -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    

                    // State H           
                    stateName = StateNameEnum.H.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action H -> D
                    stateNameNext = StateNameEnum.D.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action H -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State L           
                    stateName = StateNameEnum.L.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action L -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action L -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action L -> H
                    stateNameNext = StateNameEnum.H.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    /*
                    // State P           
                    stateName = StateNameEnum.P.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action P -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action P -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    */

                    // State K           
                    stateName = StateNameEnum.K.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action K -> L
                    stateNameNext = StateNameEnum.L.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action K -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State O           
                    stateName = StateNameEnum.O.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action O -> P
                    stateNameNext = StateNameEnum.P.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });
                    // action O -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action O -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State J           
                    stateName = StateNameEnum.J.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action J -> K
                    stateNameNext = StateNameEnum.K.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action J -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State N           
                    stateName = StateNameEnum.N.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action N -> O
                    stateNameNext = StateNameEnum.O.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action N -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State I           
                    stateName = StateNameEnum.I.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action I -> E
                    stateNameNext = StateNameEnum.E.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> M
                    stateNameNext = StateNameEnum.M.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action I -> J
                    stateNameNext = StateNameEnum.J.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State M           
                    stateName = StateNameEnum.M.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action M -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action M -> N
                    stateNameNext = StateNameEnum.N.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    // State E           
                    stateName = StateNameEnum.E.EnumToString();
                    q.AddState(state = new QState(stateName, q));
                    // action E -> A
                    stateNameNext = StateNameEnum.A.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
                    // action E -> I
                    stateNameNext = StateNameEnum.I.EnumToString();
                    state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
                    // action outcome probability
                    fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

                    q.RunTraining();
                    q.PrintQLearningStructure();
                    q.ShowPolicy();
                    #endregion
                    break;
            }
            #region code mau
            /*
            // insert the end states here, e.g. goal state
            q.EndStates.Add(StateNameEnum.H.EnumToString());


            // State A           
            stateName = StateNameEnum.A.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action A -> E
            stateNameNext = StateNameEnum.E.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State C           
            stateName = StateNameEnum.C.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action C->D
            stateNameNext = StateNameEnum.D.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State D           
            stateName = StateNameEnum.D.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action D->C
            stateNameNext = StateNameEnum.C.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action D->H
            stateNameNext = StateNameEnum.H.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = 100 });

            // State E           
            stateName = StateNameEnum.E.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action E->A
            stateNameNext = StateNameEnum.A.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action E->I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State H           
            stateName = StateNameEnum.H.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action H->D
            stateNameNext = StateNameEnum.D.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action H->L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State I           
            stateName = StateNameEnum.I.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action I->E
            stateNameNext = StateNameEnum.E.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action I->J
            stateNameNext = StateNameEnum.J.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action I->M
            stateNameNext = StateNameEnum.M.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State J           
            stateName = StateNameEnum.J.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action J->I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action J->K
            stateNameNext = StateNameEnum.K.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action J->N
            stateNameNext = StateNameEnum.N.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State K           
            stateName = StateNameEnum.K.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action K->J
            stateNameNext = StateNameEnum.J.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action K->L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action K->O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State L           
            stateName = StateNameEnum.L.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action L->K
            stateNameNext = StateNameEnum.K.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action L->H
            stateNameNext = StateNameEnum.H.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action L->P
            stateNameNext = StateNameEnum.P.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State M           
            stateName = StateNameEnum.M.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action M->I
            stateNameNext = StateNameEnum.I.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action M->N
            stateNameNext = StateNameEnum.N.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State N           
            stateName = StateNameEnum.N.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action N->M
            stateNameNext = StateNameEnum.M.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action N->J
            stateNameNext = StateNameEnum.J.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action N->O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State O           
            stateName = StateNameEnum.O.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action O->N
            stateNameNext = StateNameEnum.N.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action O->K
            stateNameNext = StateNameEnum.K.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action O->P
            stateNameNext = StateNameEnum.P.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });

            // State P           
            stateName = StateNameEnum.P.EnumToString();
            q.AddState(state = new QState(stateName, q));
            // action P->O
            stateNameNext = StateNameEnum.O.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // action P->L
            stateNameNext = StateNameEnum.L.EnumToString();
            state.AddAction(fromTo = new QAction(stateName, new QActionName(stateName, stateNameNext)));
            // action outcome probability
            fromTo.AddActionResult(new QActionResult(stateName, stateNameNext, 1.0) { Reward = -1 });
            // ----------- End Insert the path setup here -----------
            
            q.RunTraining();
            q.PrintQLearningStructure();
            q.ShowPolicy();
            */
            #endregion
        }
    }
}