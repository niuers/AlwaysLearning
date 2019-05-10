# Apache Spark
## Definition
Apache Spark is a unified analytics (computing) engine (and a set of libraries) for large-scale (parallel) data processing 
(on computer clusters).

Spark is a distributed programming model in which the user specifies transformations. 
Multiple transformations build up a directed acyclic graph of instructions. 
An action begins the process of executing that graph of instructions, as a single job, 
by breaking it down into stages and tasks to execute across the cluster. 
The logical structures that we manipulate with transformations and actions are DataFrames and Datasets. 
To create a new DataFrame or Dataset, you call a transformation. 
To start computation or convert to native language types, you call an action.

Spark is effectively a programming language of its own. 
Internally, Spark uses an engine called Catalyst that maintains its own type information through the planning and processing of work. 
Spark will convert an expression written in an input language (e.g. Python) to Spark's internal Catalyst representation of that same type 
information. It then will operate on that internal representation.

## Characteristics
1. *Unified*: It supports a wide range of data analytics tasks, simple data loading, SQL queries, machine learning and streaming computation, over the same computing engine and with a consistent set of APIs.
1. *Analytics/Computing Engine*: It's not a persistence store. Compare with Hadoop Architecture, which has HDFS as storage and MapReduce as computing engine.
1. *Libraries*: Spark Core, Spark SQL, MLlib, Spark Streaming/Structured Streaming, GraphX
1. High Performance for both batch and streaming data: DAG scheduler, a query optimizer, and a physical execution engine. Spark is scalable, massively parallel, and in-memory execution.
1. Ease of Use: Spark offers over 80 high-level operators that make it easy to build parallel apps. And you can use it interactively from the Scala, Python, R, and SQL shells.
1. Generality: SQL, Streaming, Graph and ML
1. Runs Everywhere: Spark runs on Hadoop, Apache Mesos, Kubernetes, standalone, or in the cloud. It can access diverse data sources.
1. Open Source


# Cluster Manager
The cluster manager is responsible for maintaining a cluster of machines that will run your Spark Application(s). It is responsible for keeping track of the resources available.
1. Can be one of Standalone cluster manager, YARN, or Mesos etc.
1. The manager starts a master and many worker nodes.
1. When it comes time to actually run a Spark Application, we request resources from the cluster manager to run it. 
Depending on how our application is configured, this can include a place to run the Spark driver or might be just resources for the executors for our Spark Application. 
Over the course of Spark Application execution, the cluster manager will be responsible for managing the underlying machines that our application is running on.

# Spark Application
Spark application consists of one driver process and several executor processes.

## Spark Driver Process
1. Driver process runs main() [e.g. 'ps' command will show: python3.6 my_main.py for the driver process] and is responsible for executing the driver program's commands across the executors to complete a given task. The driver process is called the SparkSession. 
1. The driver node (where the driver process is running) can be different from the master node in spark cluster.
1. Driver process is responsible for three things:
   1. maintaining information about the Spark Application
   1. responding to a user's program or input
   1. analyzing, distributing, and scheduling work across the executors
1. It is the controller of the execution of a Spark Application and maintains all of the state of the Spark cluster (the state and tasks of the executors). It must interface with the cluster manager in order to actually get physical resources and launch executors.

A Spark application can contain multiple jobs, each job could have multiple stages, and each stage has multiple tasks.


## Spark Executors
1. Executor processes are responsible for:
   1. executing code assigned to it by the driver
   1. reporting the state of the computation on that executor back to the driver node.
1. Each Worker manages one or multiple ExecutorBackend processes. Each ExecutorBackend launches and manages an Executor instance. Each Executor maintains a thread pool, in which each task runs as a thread. The tasks within the same Executor belong to the same application.
1. Spark executors are the processes that perform the tasks assigned by the Spark driver. Executors have one core responsibility: take the tasks assigned by the driver, run them, and report back their state (success or failure) and results. Each Spark Application has its own separate executor processes.

# Execution Mode
An execution mode gives you the power to determine where the aforementioned resources are physically located when you go to run your application. You have three modes to choose from:

## Cluster Mode
In cluster mode, a user submits a pre-compiled JAR, Python script, or R script to a cluster manager. 
The cluster manager then launches the driver process on a worker node inside the cluster, in addition to the executor processes. This means that the cluster manager is responsible for maintaining all Spark Application–related processes.

## Client Mode
Client mode is nearly the same as cluster mode except that the Spark driver remains on the client machine that submitted the application. This means that the client machine is responsible for maintaining the Spark driver process, and the cluster manager maintains the executor processses.
## Local Mode
It runs the entire Spark Application on a single machine. It achieves parallelism through threads on that single machine.

# The Life Cycle of Spark Application
## Step 1: Client Request
1. When you to submit an actual application, pre-compiled JAR or library. At this point, you are executing code on your local machine and you're going to make a request to the cluster manager driver node. 
1. Here, we are explicitly asking for resources for the Spark driver process only (not the cluster manager driver, i.e. master). We assume that the cluster manager accepts this offer and places the driver onto a node in the cluster. The client process that submitted the original job exits and the application is off and running on the cluster.

## Step 2: Launch
1. Now that the driver process has been placed on the cluster, it begins running user code. 
This code must include a SparkSession that initializes a Spark cluster (e.g., driver + executors). 
The SparkSession will subsequently communicate with the cluster manager (the darker line), 
asking it to launch Spark executor processes across the cluster (the lighter lines). 
The number of executors and their relevant configurations are set by the user via the command-line arguments in the original spark-submit call.
1. The cluster manager responds by launching the executor processes (assuming all goes well) and sends the relevant information about their locations to the driver process.

### Spark Session
1. Each Spark application is made up of one or more Spark jobs. Spark jobs within an application are executed serially (unless you use threading to launch multiple actions in parallel).
1. The first step of any Spark Application is creating a SparkSession.

### SparkContext
1. A SparkContext object within the SparkSession represents the connection to the Spark cluster. This class is how you communicate with some of Spark's lower-level APIs, such as RDDs.
1. After you initialize your SparkSession, it's time to execute some code. All Spark code compiles down to RDDs.

## Step 3: Execution
1. The driver and the workers communicate among themselves, executing code and moving data around. The driver schedules tasks onto each worker, and each worker responds with the status of those tasks and success or failure.
1. What you have when you call collect (or any action) is the execution of a Spark job that individually consist of stages and tasks.
1. A Spark job represents a set of transformations triggered by an individual action.
1. In general, there should be one Spark job for one action. Actions always return results. Each job breaks down into a series of stages, the number of which depends on how many shuffle operations need to take place.


### Stage
1. Stages in Spark represent groups of tasks that can be executed together to compute the same operation on multiple machines. In general, Spark will try to pack as much work as possible (i.e., as many transformations as possible inside your job) into the same stage, but the engine starts new stages after operations called shuffles. 
1. A shuffle represents a physical repartitioning of the data - for example, sorting a DataFrame, or grouping data that was loaded from a file by key (which requires sending records with the same key to the same node). This type of repartitioning requires coordinating across executors to move data around. Spark starts a new stage after each shuffle, and keeps track of what order the stages must run in to compute the final result.
1. The spark.sql.shuffle.partitions default value is 200, which means that when there is a shuffle performed during execution, it outputs 200 shuffle partitions by default. To set it use: spark.conf.set("spark.sql.shuffle.partitions", 50)
1. A good rule of thumb is that the number of partitions should be larger than the number of executors on your cluster, potentially by multiple factors depending on the workload. If you are running code on your local machine, it would behoove you to set this value lower because your local machine is unlikely to be able to execute that number of tasks in parallel. 
1. This is more of a default for a cluster in which there might be many more executor cores to use. Regardless of the number of partitions, that entire stage is computed in parallel. The final result aggregates those partitions individually, brings them all to a single partition before finally sending the final result to the driver. We'll see this configuration several times over the course of this part of the book.

### Tasks
1. Stages in Spark consist of tasks. Each task corresponds to a combination of blocks of data and a set of transformations that will run on a single executor. If there is one big partition in our dataset, we will have one task. If there are 1,000 little partitions, we will have 1,000 tasks that can be executed in parallel. A task is just a unit of computation applied to a unit of data (the partition). Partitioning your data into a greater number of partitions means that more can be executed in parallel. This is not a panacea, but it is a simple place to begin with optimization.

### Execution Details
1. Spark automatically pipelines stages and tasks that can be done together, such as a map operation followed by another map operation. Second, for all shuffle operations, Spark writes the data to stable storage (e.g., disk), and can reuse it across multiple jobs.

### Pipeline
1. An important part of what makes Spark an "in-memory computation tool" is that unlike the tools that came before it (e.g., MapReduce), Spark performs as many steps as it can at one point in time before writing data to memory or disk. One of the key optimizations that Spark performs is pipelining, which occurs at and below the RDD level. 
1. With pipelining, any sequence of operations that feed data directly into each other, without needing to move it across nodes, is collapsed into a single stage of tasks that do all the operations together. 
1. For example, if you write an RDD-based program that does a map, then a filter, then another map, these will result in a single stage of tasks that immediately read each input record, pass it through the first map, pass it through the filter, and pass it through the last mapfunction if needed. 
1. This pipelined version of the computation is much faster than writing the intermediate results to memory or disk after each step. The same kind of pipelining happens for a DataFrame or SQL computation that does a select, filter, and select.
1. From a practical point of view, pipelining will be transparent to you as you write an application - the Spark runtime will automatically do it - but you will see it if you ever inspect your application through the Spark UI or through its log files, where you will see that multiple RDD or DataFrame operations were pipelined into a single stage.

### Shuffle Persistence
1. When Spark needs to run an operation that has to move data across nodes, such as a reduce-by-key operation (where input data for each key needs to first be brought together from many nodes), the engine can't perform pipelining anymore, and instead it performs a cross-network shuffle.
1. Spark always executes shuffles by first having the "source" tasks (those sending data) write shuffle files to their local disks during their execution stage. Then, the stage that does the grouping and reduction launches and runs tasks that fetch their corresponding records from each shuffle file and performs that computation (e.g., fetches and processes the data for a specific range of keys). Saving the shuffle files to disk lets Spark run this stage later in time than the source stage (e.g., if there are not enough executors to run both at the same time), and also lets the engine re-launch reduce tasks on failure without rerunning all the input tasks.
1. One side effect you'll see for shuffle persistence is that running a new job over data that's already been shuffled does not rerun the "source" side of the shuffle. Because the shuffle files were already written to disk earlier, Spark knows that it can use them to run the later stages of the job, and it need not redo the earlier ones. In the Spark UI and logs, you will see the pre-shuffle stages marked as "skipped". This automatic optimization can save time in a workload that runs multiple jobs over the same data, but of course, for even better performance you can perform your own caching with the DataFrame or RDD cache method, which lets you control exactly which data is saved and where. You'll quickly grow accustomed to this behavior after you run some Spark actions on aggregated data and inspect them in the UI.

### Logical Instructions to Physical Plan
The execution steps of a single structured API query from user code to executed code:
1. Logic instructions: Your DataFrame/Dataset/SQL Code.
1. If valid code, Spark converts this to a Logical Plan.
This logical plan only represents a set of abstract transformations that do not refer to executors or drivers, it's purely to convert the user's set of expressions into the most optimized version. It does this by converting user code into an unresolved logical plan. This plan is unresolved because although your code might be valid, the tables or columns that it refers to might or might not exist. Spark uses the catalog, a repository of all table and DataFrame information, to resolve columns and tables in the analyzer. The analyzer might reject the unresolved logical plan if the required table or column name does not exist in the catalog. If the analyzer can resolve it, the result is passed through the Catalyst Optimizer, a collection of rules that attempt to optimize the logical plan by pushing down predicates or selections.
1. Spark (Catalyst Optimizer) transforms this Logical Plan to a Physical Plan, checking for optimizations along the way.
The physical plan, often called a Spark plan, specifies how the logical plan will execute on the cluster by generating different physical execution strategies and comparing them through a cost model. Physical planning results in a series of RDDs and transformations. This result is why you might have heard Spark referred to as a compiler - it takes queries in DataFrames, Datasets, and SQL and compiles them into RDD transformations for you.
1. Spark then executes this Physical Plan (RDD manipulations) on the cluster.
Spark performs further optimizations at runtime, generating native Java bytecode that can remove entire tasks or stages during execution. Finally the result is returned to the user.

```
# Check execution plan
pdf.explain(True): # True shows all 3 plans
```

## Step 4: Completion
1. After a Spark Application completes, the driver processs exits with either success or failure (Figure 15–7). The cluster manager then shuts down the executors in that Spark cluster for the driver. At this point, you can see the success or failure of the Spark Application by asking the cluster manager for this information.

# Spark APIs
## "Low-Level" unstructured APIs
1. RDD, Resilient Distributed Datasets: 
RDDs are lower level than DataFrames because they reveal physical execution characteristics (like partitions) to end users. 
There are basically no instances in modern Spark, for which you should be using RDDs instead of the structured APIs beyond manipulating some very raw unprocessed and unstructured data.
1. Distributed shared variables: accumulators, broadcast variables
1. SparkContext
## "Higher-Level" structured APIs
1. DataFrame: Spark maintains the types in DataFrame completely and only checks whether those types line up to those specified in the schema at runtime. Datasheets, on the other hand, check whether types conform to the specification at compile time.
1. Datasets: A type-safe structured API for writing statically typed code in Java and Scala. It's not available in dynamically typed languages like Python and R. To Spark (in Scala), DataFrames are simply Datasets of Type Row. The "Row" type is Spark's internal representation of its optimized in-memory format for computation, which can avoid the high cost associated wtih JVM type's garbage collection and object instantiation.
1. SQL Table
There is no performance difference between writing SQL queries or writing DataFrame code, they both "compile" to the same underlying plan that we specify in DataFrame code.
1. Structured Streaming

* You should generally use the lower-level APIs in three situations:
* You need some functionality that you cannot find in the higher-level APIs; for example, if you need very tight control over physical data placement across the cluster.
* You need to maintain some legacy code base written using RDDs.
* You need to do some custom shared variable manipulation. 

All Spark workloads compile down to these lower-level fundamental primitives. When you're calling a DataFrame transformation, it actually just becomes a set of RDD transformations.

## Spark DataFrame
1. Both Datasets and DataFrames are (distributed) table-like collections with well-defined rows and columns. SQL Tables and views are basically the same thing as DataFrames. We just execute SQL against them instead of DataFrame code.
1. To Spark, columns are logical constructions that simply represent a value computed on a per-record basis by means of an expression. So column is an expression. In the simplest case, expr("someCol") is equivalent to col("someCol"). This means that to have a real value for a column, we need to have a row; and to have a row, we need to have a DataFrame. You cannot manipulate an individual column outside the context of a DataFrame; you must use Spark transformations within a DataFrame to modify the contents of a column. Columns are not resolved until we compare the column names with those we are maintaining in the catalog. Column and table resolution happens in the analyzer phase. Columns and transformations of those columns compile to the same logical plan as parsed expressions.
1. In Spark, each row in a DataFrame is a single record. Spark represents this record as an object of type Row. Spark manipulates Row objects using column expressions in order to produce usable values. Row objects internally represent arrays of bytes. The byte array interface is never shown to users because we only use column expressions to manipulate them.
1. Work with column names having reserved characters, e.g. space, '-'
df.selectExpr(  "`This Long Column-Name`",    "`This Long Column-Name` as `new col`") .show(2)
dfWithLongColName.select(expr("`This Long Column-Name`")).columns
Unions are currently performed based on location, not on the schema. This means that columns will not automatically line up the way you think they might.
1. A common way to refer to a new DataFrame is to make the DataFrame into a view or register it as a table so that you can reference it more dynamically in your code.
sort and orderBy are equivalent
df.sort("count").show(5)
df.orderBy("count", "DEST_COUNTRY_NAME").show(5)
df.orderBy(col("count"), col("DEST_COUNTRY_NAME")).show(5)
1. Repartition will incur a full shuffle of the data, regardless of whether one is necessary. This means that you should typically only repartition when the future number of partitions is greater than your current number of partitions or when you are looking to partition by a set of columns. An important optimization opportunity is to partition the data according to some frequently filtered columns, which control the physical layout of data across the cluster including the partitioning scheme and the number of partitions. If you know that you're going to be filtering by a certain column often, it can be worth repartitioning based on that column.
1. Coalesce, on the other hand, will not incur a full shuffle and will try to combine partitions.
1. The method toLocalIterator collects partitions to the driver as an iterator. This method allows you to iterate over the entire dataset partition-by-partition in a serial manner:
1. You can use count to get an idea of the total size of your dataset but another common pattern is to use it to cache an entire DataFrame in memory. when performing a count(*), Spark will count null values (including rows containing all nulls). However, when counting an individual column, Spark will not count the null values. Counting is a bit of a special case because it exists as a method. For this, usually we prefer to use the countfunction.
from pyspark.sql.functions import count
 df.groupBy("InvoiceNo").agg(
    count("Quantity").alias("quan"),
    expr("count(Quantity)")).show()
Collect unique values into a set or list
from pyspark.sql.functions import collect_set, collect_list
df.agg(collect_set("Country"), collect_list("Country")).show()
1. Window function
from pyspark.sql.functions import max
maxPurchaseQuantity = max(col("Quantity")).over(windowSpec)
1. One "gotcha" that can come up is if you're working with null data when creating Boolean expressions. If there is a null in your data, you'll need to treat things a bit differently. Here's how you can ensure that you perform a null-safe equivalence test:
df.where(col("Description").eqNullSafe("hello")).show()
1. When you define a schema in which all columns are declared to not have null values, Spark will not enforce that and will happily let null values into that column. The nullable signal is simply to help Spark SQL optimize for handling that column.

## Spark Dataset
1. if you use Scala or Java, all "DataFrames" are actually Datasets of type Row.
1. For example, given a class Person with two fields, name (string) and age (int), an encoder directs Spark to generate code at runtime to serialize the Person object into a binary structure. When using DataFrames or the "standard" Structured APIs, this binary structure will be a Row. When we want to create our own domain-specific objects, we specify a case class in Scala or a JavaBean in Java. Spark will allow us to manipulate this object (in place of a Row) in a distributed manner.
1. When you use the Dataset API, for every row it touches, this domain specifies type, Spark converts the Spark Row format to the object you specified (a case class or Java class). This conversion slows down your operations but can provide more flexibility.
1. You might ponder, if I am going to pay a performance penalty when I use Datasets, why should I use them at all? If we had to condense this down into a canonical list, here are a couple of reasons:
1. When the operation(s) you would like to perform cannot be expressed using DataFrame manipulations
1. When you want or need type-safety, and you're willing to accept the cost of performance to achieve it


## Spark Resilient Distributed Dataset (RDD)
1. An RDD represents an immutable, partitioned collection of records that can be operated on in parallel. Unlike DataFrames though, where each record is a structured row containing fields with a known schema, in RDDs the records are just Java, Scala, or Python objects of the programmer's choosing. You can store anything you want in these objects, in any format you want.
1. Every manipulation and interaction between values must be defined by hand, meaning that you must "reinvent the wheel" for whatever task you are trying to carry out. This applies to optimization as well.
1. You will likely only be creating two types of RDDs: the "generic" RDD type or a key-value RDD that provides additional functions, such as aggregating by key.
1. Internally, each RDD is characterized by five main properties:
* A list of partitions
* A function for computing each split
* A list of dependencies on other RDDs
* Optionally, a Partitioner for key-value RDDs (e.g., to say that the RDD is hash-partitioned)
* Optionally, a list of preferred locations on which to compute each split (e.g., block locations for a Hadoop Distributed File System [HDFS] file) These properties determine all of Spark's ability to schedule and execute the user program.
1. However, there is no concept of "rows" in RDDs; individual records are just raw Java/Scala/Python objects, and you manipulate those manually instead of tapping into the repository of functions that you have in the structured APIs.
1. For Scala and Java, the performance is for the most part the same, the large costs incurred in manipulating the raw objects.
1. Python, however, can lose a substantial amount of performance when using RDDs. Running Python RDDs equates to running Python user-defined functions (UDFs) row by row. We serialize the data to the Python process, operate on it in Python, and then serialize it back to the Java Virtual Machine (JVM). This causes a high overhead for Python RDD manipulations. 
1. The fundamental issue in the code below is that each executor must hold all values for a given key in memory before applying the function to them.

`KVcharacters.groupByKey().map(lambda row: (row[0], reduce(addFunc, row[1]))).collect()
The reduceByKey method returns an RDD of a group (the key) and sequence of elements that are not guranteed to have an ordering. Therefore this method is completely appropriate when our workload is associative but inappropriate when the order matters.
1. Controlling Partitions

With RDDs, you have control over how data is exactly physically distributed across the cluster. Some of these methods are basically the same from what we have in the Structured APIs but the key addition (that does not exist in the Structured APIs) is the ability to specify a partitioning function (formally a custom Partitioner).
coalesce effectively collapses partitions on the same worker in order to avoid a (full) shuffle of the data when repartitioning. For instance, our words RDD is currently two partitions, we can collapse that to one partition by using coalesce without bringing about a shuffle of the data:
coalesce uses existing partitions to minimize the amount of data that's shuffled. repartition creates new partitions and does a full shuffle. coalesce results in partitions with different amounts of data (sometimes partitions that have much different sizes) and repartition results in roughly equal sized partitions.
The repartition operation allows you to repartition your data up or down but performs a shuffle across nodes in the process.
1. Custom Partition

This ability is one of the primary reasons you'd want to use RDDs. In short, the sole goal of custom partitioning is to even out the distribution of your data across the cluster so that you can work around problems like data skew.
At times, you will need to perform some very low-level partitioning because you're working with very large data and large key skew.
Any object that you hope to parallelize (or function) must be serializable

## Broadcast Variables
1. Broadcast variables are a way you can share an immutable value efficiently around the cluster without encapsulating that variable in a function closure. The normal way to use a variable in your driver node inside your tasks is to simply reference it in your function closures (e.g., in a map operation), but this can be inefficient, especially for large variables such as a lookup table or a machine learning model. The reason for this is that when you use a variable in a closure, it must be deserialized on the worker nodes many times (one per task). Moreover, if you use the same variable in multiple Spark actions and jobs, it will be re-sent to the workers with every job instead of once.
1. This is where broadcast variables come in. Broadcast variables are shared, immutable variables that are cached on every machine in the cluster instead of serialized with every single task. The canonical use case is to pass around a large lookup table that fits in memory on the executors and use that in a function.
1. We reference broadcast variable via the value method, which returns the exact value that we had earlier. This method is accessible within serialized functions without having to serialize the data. This can save you a great deal of serialization and deserialization costs because Spark transfers data more efficiently around the cluster using broadcasts:
1. The only difference between this and passing it into the closure is that we have done this in a much more efficient manner (Naturally, this depends on the amount of data and the number of executors. For very small data (low KBs) on small clusters, it might not be). Although this small dictionary probably is not too large of a cost, if you have a much larger value, the cost of serializing the data for every task can be quite significant.

## Accumulator
1. Spark's second type of shared variable, are a way of updating a value inside of a variety of transformations and propagating that value to the driver node in an efficient and fault-tolerant way.
1. Accumulators provide a mutable variable that a Spark cluster can safely update on a per-row basis. You can use these for debugging purposes (say to track the values of a certain variable per partition in order to intelligently use it over time) or to create low-level aggregation. Accumulators are variables that are "added" to only through an associative and commutative operation and can therefore be efficiently supported in parallel. You can use them to implement counters (as in MapReduce) or sums. Spark natively supports accumulators of numeric types, and programmers can add support for new types.


## Cache vs Persistence vs. Checkpoint
Checkpointing is the act of saving an RDD to disk so that future references to this RDD point to those intermediate partitions on disk rather than recomputing the RDD from its original source. This is similar to caching except that it's not stored in memory, only disk.

## Pipe
1. With pipe, you can return an RDD created by piping elements to a forked external process. The resulting RDD is computed by executing the given process once per partition. All elements of each input partition are written to a process's stdin as lines of input separated by a newline. The resulting partition consists of the process's stdout output, with each line of stdout resulting in one element of the output partition. A process is invoked even for empty partitions.
1. Spark operates on a per-partition basis when it comes to actually executing code.

# Spark Join
## Two Communication Strategies
Spark approaches cluster communication in two different ways during joins. 
One thing is important to consider is if you partition your data correctly prior to a join, you can end up with much more efficient execution because even if a shuffle is planned, if data from two different DataFrames is already located on the same machine, Spark can avoid the shuffle.
### Shuffle Join

Shuffle join results in an all-to-all communication. When you join a big table to another big table, you end up with a shuffle join. 
In a shuffle join, every worker node ( and potentially every partition) talks to every other node and they share data according to which node has a certain key or set of keys (on which you are joining) during the entire join process (with no intelligent partitioning of data). These joins are expensive because the network can become congested with traffic, especially if your data is not partitioned well.
### Broadcast Join

When we join big table with small table, broadcast join strategy should be used. We will replicate our small DataFrame onto every worker node in the cluster (be it located on one machine or many).
What this does is prevent us from performing the all-to-all communication during the entire join process. Instead, we perform it only once at the beginning and then let each individual worker node perform the work without having to wait or communicate with any other worker node.






# Transformations
## Transformations with narrow dependencies
Each input partition will contribute to only one output partition (1–1).
With narrow transformations, Spark will automatically perform an operation called pipelining, meaning that if we specify multiple filters on DataFrames, they'll all be performed in-memory. The same cannot be said for shuffles (Wide Transformations).
## Transformations with wide dependencies (Shuffle)
One Input partition contributes to many output partitions (1-n).
Reading data is a transformation, and is therefore a lazy operation.

# Spark SQL
## Hive
1. Before Spark's rise, Hive was the de facto big data SQL access layer.
1. Spark SQL is intended to operate as an online analytic processing (OLAP) database, not an online transaction processing (OLTP) database. This means that it is not intended to perform extremely low-latency queries. Even though support for in-place modifications is sure to be something that comes up in the future, it's not something that is currently available.
1. The method sql on the SparkSession object returns a DataFrame. This is an immensely powerful interface because there are some transformations that are much simpler to express in SQL code than in DataFrames.
1. The highest level abstraction in Spark SQL is the Catalog. The Catalog is an abstraction for the storage of metadata about the data stored in your tables as well as other helpful things like databases, tables, functions, and views.
1. Tables are logically equivalent to a DataFrame in that they are a structure of data against which you run commands. The core difference between tables and DataFrames is this: you define DataFrames in the scope of a programming language, whereas you define tables within a database. This means that when you create a table (assuming you never changed the database), it will belong to the default database. An important thing to note is that tables always contain data. There is no notion of a temporary table, only a view, which does not contain data. This is important because if you go to drop a table, you can risk losing the data when doing so.
1. One important note is the concept of managed versus unmanaged tables. Tables store two important pieces of information. The data within the tables as well as the data about the tables; that is, the metadata. You can have Spark manage the metadata for a set of files as well as for the data. When you define a table from files on disk, you are defining an unmanaged table. When you use saveAsTable on a DataFrame, you are creating a managed table for which Spark will track of all of the relevant information. This will read your table and write it out to a new location in Spark format. You can see this reflected in the new explain plan.

## Views
1. A view specifies a set of transformations on top of an existing table - basically just saved query plans, which can be convenient for organizing or reusing your query logic. Spark has several different notions of views. Views can be global, set to a database, or per session.
1. Like tables, you can create temporary views that are available only during the current session and are not registered to a database
1. A view is effectively a transformation and Spark will perform it only at query time. This means that it will only apply that filter after you actually go to query the table (and not earlier). Effectively, views are equivalent to creating a new DataFrame from an existing DataFrame.

# Actions
1. Actions to view data in the console
1. Actions to collect data to native objects in the respective language
1. Actions to write to output data sources


## Customize Data Partition
Spark does partition on your data automatically, but not always in the way you would like to do. For example, you have the monthly time series of security prices for past one year. You may want to partition the data based on the month, so you'll have 12 partitions and you can run some analytics in parallel on each month separately. 
The default partition provided by Spark won't work this way. It may partition your data evenly. But most likely you won't have same amount of data in each month. 
How can we do this then? 
Suppose myrecords is a Spark DataFrame that holds all my data. And suppose there's a column called 'DATE'. Let's work with pySpark.
1. type(myrecords)
pyspark.sql.dataframe.DataFrame
2. Check number of partitions
myrecords.rdd.getNumPartitions()
3. The repartition() won't work as expected here
records = myrecords.repartition("DATE")
records.rdd.getNumPartitions()
200 #Default partition number, it may vary for you
4. Let's check what partitioner we are using
print(records.rdd.partitioner)
None
#So we haven't specified a partitioner. Let's define one.
5. def date_partitioner(d):
       return hash(d.month)
6. Now let's use it to partition
from pyspark import SparkContext
df = records.rdd.map(lambda el: (el['DATE'], el)) \
            .partitionBy(12, date_partitioner)
df.getNumPartitions()
12
7. Check how many data in each partition
df.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
It should show you the accurate data in each month. There'll be no overlap of a month's data in different partitions (as generated by the default spark partitioner)
References:
1. [Partitioning in Apache Spark](https://medium.com/parrot-prediction/partitioning-in-apache-spark-8134ad840b0)

## Repartition
https://medium.com/r/?url=https%3A%2F%2Fdatascience.stackexchange.com%2Fquestions%2F28177%2Fpyspark-dataframe-repartition
The default value for spark.sql.shuffle.partitions is 200, and configures the number of partitions that are used when shuffling data for joins or aggregations.
dataframe.repartition('id') creates 200 partitions with ID partitioned based on Hash Partitioner. Dataframe Row's with the same ID always goes to the same partition. If there is DataSkew on some ID's, you'll end up with inconsistently sized partitions.
1. Get number of rows in each partition
new_data = df.rdd.glom().map(len).collect()

## Efficient UDF with PySpark
https://medium.com/r/?url=https%3A%2F%2Fflorianwilhelm.info%2F2017%2F10%2Fefficient_udfs_with_pyspark%2F

# UDF
If you have created the UDF functions, we need to register them with Spark so that we can use them on all of our worker machines. Spark will serialize the function on the driver and transfer it over the network to all executor processes. This happens regardless of language.
When you use the function, there are essentially two different things that occur. If the function is written in Scala or Java, you can use it within the Java Virtual Machine (JVM). This means that there will be little performance penalty aside from the fact that you can't take advantage of code generation capabilities that Spark has for built-in functions. There can be performance issues if you create or use a lot of objects.
If the function is written in Python, something quite different happens. Spark starts a Python process on the worker, serializes all of the data to a format that Python can understand (remember, it was in the JVM earlier), executes the function row by row on that data in the Python process, and then finally returns the results of the row operations to the JVM and Spark. 
Starting this Python process is expensive, but the real cost is in serializing the data to Python. This is costly for two reasons: it is an expensive computation, but also, after the data enters Python, Spark cannot manage the memory of the worker. This means that you could potentially cause a worker to fail if it becomes resource constrained (because both the JVM and Python are competing for memory on the same machine). We recommend that you write your UDFs in Scala or Java - the small amount of time it should take you to write the function in Scala will always yield significant speed ups, and on top of that, you can still use the function from Python!

## Predicate Pushdown
predicate pushdown on DataFrames. If we build a large Spark job but specify a filter at the end that only requires us to fetch one row from our source data, the most efficient way to execute this is to access the single record that we need. Spark will actually optimize this for us by pushing the filter down automatically.



# Spark Data Sources API
## Core structure to read

DataFrameReader.format(…).option("key", "value").schema(…).load()
## Core structure to write
DataFrameWriter.format(…).option(…).partitionBy(…).bucketBy(…).sotBy( …).save()

## Splittable File Types and Compression
Certain file formats are fundamentally "splittable." This can improve speed because it makes it possible for Spark to avoid reading an entire file, and access only the parts of the file necessary to satisfy your query. 
Additionally if you're using something like Hadoop Distributed File System (HDFS), splitting a file can provide further optimization if that file spans multiple blocks.
 In conjunction with this is a need to manage compression. Not all compression schemes are splittable. How you store your data is of immense consequence when it comes to making your Spark jobs run smoothly. We recommend Parquet with gzip compression.
### Reading Data in Parallel
Multiple executors cannot read from the same file at the same time necessarily, but they can read different files at the same time. 
In general, this means that when you read from a folder with multiple files in it, each one of those files will become a partition in your DataFrame and be read in by available executors in parallel (with the remaining queueing up behind the others).
### Writing Data in Parallel
The number of files or data written is dependent on the number of partitions the DataFrame has at the time you write out the data. By default, one file is written per partition of the data. This means that although we specify a "file," it's actually a number of files within a folder, with the name of the specified file, with one file per each partition that is written.
### PARTITIONING
Partitioning is a tool that allows you to control what data is stored (and where) as you write it. When you write a file to a partitioned directory (or table), you basically encode a column as a folder. What this allows you to do is skip lots of data when you go to read it in later, allowing you to read in only the data relevant to your problem instead of having to scan the complete dataset. These are supported for all file-based data sources.
### BUCKETING
Bucketing is another file organization approach with which you can control the data that is specifically written to each file. This can help avoid shuffles later when you go to read the data because data with the same bucket ID will all be grouped together into one physical partition. This means that the data is prepartitioned according to how you expect to use that data later on, meaning you can avoid expensive shuffles when joining or aggregating.
Rather than partitioning on a specific column (which might write out a ton of directories), it's probably worthwhile to explore bucketing the data instead. This will create a certain number of files and organize our data into those "buckets". Bucketing is supported only for Spark-managed tables.
### Managing File Size
Managing file sizes is an important factor not so much for writing data but reading it later on. When you're writing lots of small files, there's a significant metadata overhead that you incur managing all of those files. Spark especially does not do well with small files, although many file systems (like HDFS) don't handle lots of small files well, either. You might hear this referred to as the "small file problem." The opposite is also true: you don't want files that are too large either, because it becomes inefficient to have to read entire blocks of data when you need only a few rows.
### Parquet File Format
1. Parquet is an open source column-oriented data store (while CSV is row-oriented format) that provides a variety of storage optimizations, especially for analytics workloads. 
1. Typically row-oriented formats are more efficient for queries that either must access most of the columns, or only read a fraction of the rows. Column-oriented formats, on the other hand, are usually more efficient for queries that need to read most of the rows, but only have to access a fraction of the columns. Analytical queries typically fall in the latter category, while transactional queries are more often in the first category.
1. It provides columnar compression, which saves storage space and allows for reading individual columns instead of entire files. 
1. CSV is a text-based format, which can not be parsed as efficiently as a binary format. This makes CSV even slower. A typical column-oriented format on the other hand is not only binary, but also allows more efficient compression, which leads to smaller disk usage and faster access. I recommend reading the Introduction section of The Design and Implementation of Modern Column-Oriented Database Systems.
1. Parquet is self-describing because it embeds schema when storing data. This results in a file that is optimized for query performance and minimizing I/O.
1. Apache Parquet is built to support very efficient compression and encoding schemes.
1. It is a file format that works exceptionally well with Apache Spark and is in fact the default file format. 
1. It supports complex types. This means that if your column is an array (which would fail with a CSV file, for example), map, or struct, you'll still be able to read and write that file without issue. 
1. Be careful when you write out Parquet files with different versions of Spark (especially older ones) because this can cause significant headache.
1. It is recommended to write data out to Parquet for long-term storage because reading from a Parquet file will always be more efficient than JSON or CSV.
1. CSV is splittable when it is a raw, uncompressed file or using a splittable compression format such as BZIP2 or LZO (note: LZO needs to be indexed to be splittable!).
1. CSV should generally be the fastest to write (???), JSON the easiest for a human to understand and Parquet the fastest to read.
1. Parquet is optimized for the Write Once Read Many (WORM) paradigm. It's slow to write, but incredibly fast to read, especially when you're only accessing a subset of the total columns. For use cases requiring operating on entire columns (???) of data, a format like CSV, JSON or even AVRO should be used.
1. If you use 'inferSchema' while loading a csv file, in order to determine with certainty the proper data types to assign to each column, Spark has to READ AND PARSE THE ENTIRE DATASET. This can be a very high cost, especially when the number of files/rows/columns is large. It also does no processing while it's inferring the schema, so if you thought it would be running your actual transformation code while it's inferring the schema, sorry, it won't. Spark has to therefore read your file(s) TWICE instead of ONCE.

# Testing in Spark
1. you should almost always favor running in cluster mode (or in client mode on the cluster itself) to reduce latency between the executors and the driver.

## Spark provides three locations to configure the system:

### Spark properties control most application parameters and can be set by using a SparkConf object
The SparkConf manages all of our application configurations. You create one via the import statement, as shown in the example that follows. After you create it, the SparkConf is immutable for that specific Spark Application. 

### Java system properties

### Hardcoded configuration files

1. Within a given Spark Application, multiple parallel jobs can run simultaneously if they were submitted from separate threads. By job, in this section, we mean a Spark action and any tasks that need to run to evaluate that action. Spark’s scheduler is fully thread-safe and supports this use case to enable applications that serve multiple requests (e.g., queries for multiple users).

By default, Spark’s scheduler runs jobs in FIFO fashion. If the jobs at the head of the queue don’t need to use the entire cluster, later jobs can begin to run right away, but if the jobs at the head of the queue are large, later jobs might be delayed significantly.

It is also possible to configure fair sharing between jobs. Under fair sharing, Spark assigns tasks between jobs in a round-robin fashion so that all jobs get a roughly equal share of cluster resources. This means that short jobs submitted while a long job is running can begin receiving resources right away and still achieve good response times without waiting for the long job to finish. This mode is best for multiuser settings.

1. If you are going to deploy on-premises, the best way to combat the resource utilization problem is to use a cluster manager that allows you to run many Spark applications and dynamically reassign resources between them, or even allows non-Spark applications on the same cluster. All of Spark’s supported cluster managers allow multiple concurrent applications, but YARN and Mesos have better support for dynamic sharing and also additionally support non-Spark workloads. 

## Standalone Mode
Spark’s standalone cluster manager is a lightweight platform built specifically for Apache Spark workloads. Using it, you can run multiple Spark Applications on the same cluster. It also provides simple interfaces for doing so but can scale to large Spark workloads. The main disadvantage of the standalone mode is that it’s more limited than the other cluster managers—in particular, your cluster can only run Spark. It’s probably the best starting point if you just want to quickly get Spark running on a cluster, however, and you do not have experience using YARN or Mesos.

## Spark on YARN
Hadoop YARN is a framework for job scheduling and cluster resource management. Even though Spark is often (mis)classified as a part of the “Hadoop Ecosystem,” in reality, Spark has little to do with Hadoop.

The number of knobs is naturally larger than that of Spark’s standalone mode because Hadoop YARN is a generic scheduler for a large number of different execution frameworks.

cluster mode has the spark driver as a process managed by the YARN cluster, and the client can exit after creating the application. In client mode, the driver will run in the client process and therefore YARN will be responsible only for granting executor resources to the application, not maintaining the master node. Also of note is that in cluster mode, Spark doesn’t necessarily run on the same machine on which you’re executing. Therefore libraries and external jars must be distributed manually or through the --jars command-line argument.

## Spark on Mesos
Apache Mesos is another clustering system that Spark can run on. In the Mesos project’s own words:

Apache Mesos abstracts CPU, memory, storage, and other compute resources away from machines (physical or virtual), enabling fault-tolerant and elastic distributed systems to easily be built and run effectively.

For the most part, Mesos intends to be a datacenter scale-cluster manager that manages not just short-lived applications like Spark, but long-running applications like web applications or other resource interfaces. Mesos is the heaviest-weight cluster manager, simply because you might choose this cluster manager only if your organization already has a large-scale deployment of Mesos, but it makes for a good cluster manager nonetheless.

## Application Scheduling
Spark has several facilities for scheduling resources between computations. First, recall that, as described earlier in the book, each Spark Application runs an independent set of executor processes. Cluster managers provide the facilities for scheduling across Spark applications. Second, within each Spark application, multiple jobs (i.e., Spark actions) may be running concurrently if they were submitted by different threads. This is common if your application is serving requests over the network. Spark includes a fair scheduler to schedule resources within each application. We introduced this topic in the previous chapter.


## DYNAMIC ALLOCATION
If you would like to run multiple Spark Applications on the same cluster, Spark provides a mechanism to dynamically adjust the resources your application occupies based on the workload. This means that your application can give resources back to the cluster if they are no longer used, and request them again later when there is demand.

This feature is disabled by default and available on all coarse-grained cluster managers; that is, standalone mode, YARN mode, and Mesos coarse-grained mode. 


One of the more important considerations is the number and type of applications you intend to be running. For instance, YARN is great for HDFS-based applications but is not commonly used for much else. Additionally, it’s not well designed to support the cloud, because it expects information to be available on HDFS. Also, compute and storage is largely coupled together, meaning that scaling your cluster involves scaling both storage and compute instead of just one or the other. Mesos does improve on this a bit conceptually, and it supports a wide range of application types, but it still requires pre-provisioning machines and, in some sense, requires buy-in at a much larger scale. For instance, it doesn’t really make sense to have a Mesos cluster for only running Spark Applications. Spark standalone mode is the lightest-weight cluster manager and is relatively simple to understand and take advantage of, but then you’re going to be building more application management infrastructure that you could get much more easily by using YARN or Mesos.

Typically Spark stores shuffle blocks (shuffle output) on a local disk on that particular node.


# Job Monitoring
## JVM
Spark runs the executors in individual Java Virtual Machines (JVMs). Therefore, the next level of detail would be to monitor the individual virtual machines (VMs) to better understand how your code is running. JVM utilities such as jstack for providing stack traces, jmap for creating heap-dumps, jstat for reporting time–series statistics, and jconsole for visually exploring various JVM properties are useful for those comfortable with JVM internals. You can also use a tool like jvisualvm to help profile Spark jobs.

There are two main things you will want to monitor: the processes running your application (at the level of CPU usage, memory usage, etc.), and the query execution inside it (e.g., jobs and tasks).

## Driver and Executor Processes
When you’re monitoring a Spark application, you’re definitely going to want to keep an eye on the driver. This is where all of the state of your application lives, and you’ll need to be sure it’s running in a stable manner. To help with this challenge, Spark has a configurable metrics system based on the Dropwizard Metrics Library. The metrics system is configured via a configuration file that Spark expects to be present at $SPARK_HOME/conf/metrics.properties. A custom file location can be specified by changing the spark.metrics.conf configuration property. 

## Queries, Jobs, Stages, and Tasks

## Spark Logs
One challenge, however, is that Python won’t be able to integrate directly with Spark’s Java-based logging library. Using Python’s logging module or even simple print statements will still print the results to standard error, however, and make them easy to find.



1. Spark performs an aggregation for each partition (in this case a hash-based aggregation) before shuffling the data around in preparation for the final stage.

The first stage has eight tasks. CSV files are splittable, and Spark broke up the work to be distributed relatively evenly between the different cores on the machine. This happens at the cluster level and points to an important optimization: how you store your files. The following stage has two tasks because we explicitly called a repartition to move the data into two partitions. The last stage has 200 tasks because the default shuffle partitions value is 200.

## Deal with Stragglers
Slow tasks are often called “stragglers.” There are many reasons they may occur, but most often the source of this issue is that your data is partitioned unevenly into DataFrame or RDD partitions. When this happens, some executors might need to work on much larger amounts of work than others. One particularly common case is that you use a group-by-key operation and one of the keys just has more data than others. In this case, when you look at the Spark UI, you might see that the shuffle data for some nodes is much larger than for others.

1. Try increasing the number of partitions to have less data per partition.

1. Try repartitioning by another combination of columns. For example, stragglers can come up when you partition by a skewed ID column, or a column where many values are null. In the latter case, it might make sense to first filter out the null values.

1. Try increasing the memory allocated to your executors if possible.

1. Monitor the executor that is having trouble and see if it is the same machine across jobs; you might also have an unhealthy executor or machine in your cluster—for example, one whose disk is nearly full.

1. If this issue is associated with a join or an aggregation, see “Slow Joins” or “Slow Aggregations”.

1. Check whether your user-defined functions (UDFs) are wasteful in their object allocation or business logic. Try to convert them to DataFrame code if possible.

1. Ensure that your UDFs or User-Defined Aggregate Functions (UDAFs) are running on a small enough batch of data. Often times an aggregation can pull a lot of data into memory for a common key, leading to that executor having to do a lot more work than others.

1. Turning on speculation, which we discuss in “Slow Reads and Writes”, will have Spark run a second copy of tasks that are extremely slow. This can be helpful if the issue is due to a faulty node because the task will get to run on a faster one. Speculation does come at a cost, however, because it consumes additional resources. In addition, for some storage systems that use eventual consistency, you could end up with duplicate output data if your writes are not idempotent. (We discussed speculation configurations in Chapter 17.)

1. Another common issue can arise when you’re working with Datasets. Because Datasets perform a lot of object instantiation to convert records to Java objects for UDFs, they can cause a lot of garbage collection. If you’re using Datasets, look at the garbage collection metrics in the Spark UI to see if they’re consistent with the slow tasks.

1. Stragglers can be one of the most difficult issues to debug, simply because there are so many possible causes. However, in all likelihood, the cause will be some kind of data skew, so definitely begin by checking the Spark UI for imbalanced amounts of data across tasks.


## Slow Aggregations

# SIGNS AND SYMPTOMS
*. Slow tasks during a groupBy call.

*. Jobs after the aggregation are slow, as well.

# POTENTIAL TREATMENTS
1. Unfortunately, this issue can’t always be solved. Sometimes, the data in your job just has some skewed keys, and the operation you want to run on them needs to be slow.

1. Increasing the number of partitions, prior to an aggregation, might help by reducing the number of different keys processed in each task.

1. Increasing executor memory can help alleviate this issue, as well. If a single key has lots of data, this will allow its executor to spill to disk less often and finish faster, although it may still be much slower than executors processing other keys.

1. If you find that tasks after the aggregation are also slow, this means that your dataset might have remained unbalanced after the aggregation. Try inserting a repartition call to partition it randomly.

1. Ensuring that all filters and SELECT statements that can be are above the aggregation can help to ensure that you’re working only on the data that you need to be working on and nothing else. Spark’s query optimizer will automatically do this for the structured APIs.

1. Ensure null values are represented correctly (using Spark’s concept of null) and not as some default value like " " or "EMPTY". Spark often optimizes for skipping nulls early in the job when possible, but it can’t do so for your own placeholder values.

1. Some aggregation functions are also just inherently slower than others. For instance, collect_list and collect_set are very slow aggregation functions because they must return all the matching objects to the driver, and should be avoided in performance-critical code.

## Slow Joins
Joins and aggregations are both shuffles, so they share some of the same general symptoms as well as treatments.

### SIGNS AND SYMPTOMS
1. A join stage seems to be taking a long time. This can be one task or many tasks.

1. Stages before and after the join seem to be operating normally.

### POTENTIAL TREATMENTS
1. Many joins can be optimized (manually or automatically) to other types of joins. 

1. Experimenting with different join orderings can really help speed up jobs, especially if some of those joins filter out a large amount of data; do those first.

1. Partitioning a dataset prior to joining can be very helpful for reducing data movement across the cluster, especially if the same dataset will be used in multiple join operations. It’s worth experimenting with different prejoin partitioning. Keep in mind, again, that this isn’t “free” and does come at the cost of a shuffle.

1. Slow joins can also be caused by data skew. There’s not always a lot you can do here, but sizing up the Spark application and/or increasing the size of executors can help, as described in earlier sections.

1. Ensuring that all filters and select statements that can be are above the join can help to ensure that you’re working only on the data that you need for the join.

1. Ensure that null values are handled correctly (that you’re using null) and not some default value like " " or "EMPTY", as with aggregations.

1. Sometimes Spark can’t properly plan for a broadcast join if it doesn’t know any statistics about the input DataFrame or table. If you know that one of the tables that you are joining is small, you can try to force a broadcast, or use Spark’s statistics collection commands to let it analyze the table.

## Slow Reads and Writes
Slow I/O can be difficult to diagnose, especially with networked file systems.

### SIGNS AND SYMPTOMS
1. Slow reading of data from a distributed file system or external system.

1. Slow writes from network file systems or blob storage.

### POTENTIAL TREATMENTS
1. Turning on speculation (set spark.speculation to true) can help with slow reads and writes. This will launch additional tasks with the same operation in an attempt to see whether it’s just some transient issue in the first task. Speculation is a powerful tool and works well with consistent file systems. However, it can cause duplicate data writes with some eventually consistent cloud services, such as Amazon S3, so check whether it is supported by the storage system connector you are using.

1. Ensuring sufficient network connectivity can be important—your Spark cluster may simply not have enough total network bandwidth to get to your storage system.

1. For distributed file systems such as HDFS running on the same nodes as Spark, make sure Spark sees the same hostnames for nodes as the file system. This will enable Spark to do locality-aware scheduling, which you will be able to see in the “locality” column in the Spark UI. 

## Driver OutOfMemoryError or Driver Unresponsive
This is usually a pretty serious issue because it will crash your Spark Application. It often happens due to collecting too much data back to the driver, making it run out of memory.

### SIGNS AND SYMPTOMS
1. Spark Application is unresponsive or crashed.

1.. OutOfMemoryErrors or garbage collection messages in the driver logs.

1. Commands take a very long time to run or don’t run at all.

1. Interactivity is very low or non-existent.

1. Memory usage is high for the driver JVM.

### POTENTIAL TREATMENTS

1. Your code might have tried to collect an overly large dataset to the driver node using operations such as collect.

1. You might be using a broadcast join where the data to be broadcast is too big. Use Spark’s maximum broadcast join configuration to better control the size it will broadcast.

1. A long-running application generated a large number of objects on the driver and is unable to release them. Java’s jmap tool can be useful to see what objects are filling most of the memory of your driver JVM by printing a histogram of the heap. However, take note that jmap will pause that JVM while running.

1. Increase the driver’s memory allocation if possible to let it work with more data.

1. Issues with JVMs running out of memory can happen if you are using another language binding, such as Python, due to data conversion between the two requiring too much memory in the JVM. Try to see whether your issue is specific to your chosen language and bring back less data to the driver node, or write it to a file instead of bringing it back as in-memory objects.

1. If you are sharing a SparkContext with other users (e.g., through the SQL JDBC server and some notebook environments), ensure that people aren’t trying to do something that might be causing large amounts of memory allocation in the driver (like working overly large arrays in their code or collecting large datasets).

## Executor OutOfMemoryError or Executor Unresponsive
Spark applications can sometimes recover from this automatically, depending on the true underlying issue.

### SIGNS AND SYMPTOMS
1. OutOfMemoryErrors or garbage collection messages in the executor logs. You can find these in the Spark UI.

1. Executors that crash or become unresponsive.

1. Slow tasks on certain nodes that never seem to recover.

### POTENTIAL TREATMENTS
1. Try increasing the memory available to executors and the number of executors.

1. Try increasing PySpark worker size via the relevant Python configurations.

1. Look for garbage collection error messages in the executor logs. Some of the tasks that are running, especially if you’re using UDFs, can be creating lots of objects that need to be garbage collected. Repartition your data to increase parallelism, reduce the amount of records per task, and ensure that all executors are getting the same amount of work.

1. Ensure that null values are handled correctly (that you’re using null) and not some default value like " " or "EMPTY", as we discussed earlier.

1. This is more likely to happen with RDDs or with Datasets because of object instantiations. Try using fewer UDFs and more of Spark’s structured operations when possible.

1. Use Java monitoring tools such as jmap to get a histogram of heap memory usage on your executors, and see which classes are taking up the most space.

1. If executors are being placed on nodes that also have other workloads running on them, such as a key-value store, try to isolate your Spark jobs from other jobs.

## Unexpected Nulls in Results
### SIGNS AND SYMPTOMS
1. Unexpected null values after transformations.

1. Scheduled production jobs that used to work no longer work, or no longer produce the right results.

### POTENTIAL TREATMENTS
1. It’s possible that your data format has changed without adjusting your business logic. This means that code that worked before is no longer valid.

1. Use an accumulator to try to count records or certain types, as well as parsing or processing errors where you skip a record. This can be helpful because you might think that you’re parsing data of a certain format, but some of the data doesn’t. Most often, users will place the accumulator in a UDF when they are parsing their raw data into a more controlled format and perform the counts there. This allows you to count valid and invalid records and then operate accordingly after the fact.

1. Ensure that your transformations actually result in valid query plans. Spark SQL sometimes does implicit type coercions that can cause confusing results. For instance, the SQL expression SELECT 5*"23" results in 115 because the string “25” converts to an the value 25 as an integer, but the expression SELECT 5 * " " results in null because casting the empty string to an integer gives null. Make sure that your intermediate datasets have the schema you expect them to (try using printSchema on them), and look for any CAST operations in the final query plan.

## No Space Left on Disk Errors
### SIGNS AND SYMPTOMS
1. You see “no space left on disk” errors and your jobs fail.

### POTENTIAL TREATMENTS
1. The easiest way to alleviate this, of course, is to add more disk space. You can do this by sizing up the nodes that you’re working on or attaching external storage in a cloud environment.

1. If you have a cluster with limited storage space, some nodes may run out first due to skew. Repartitioning the data as described earlier may help here.

1. There are also a number of storage configurations with which you can experiment. Some of these determine how long logs should be kept on the machine before being removed. For more information.

1. Try manually removing some old log files or old shuffle files from the machine(s) in question. This can help alleviate some of the issue although obviously it’s not a permanent fix.

## Serialization Errors
### SIGNS AND SYMPTOMS
1. You see serialization errors and your jobs fail.

### POTENTIAL TREATMENTS
1. This is very uncommon when working with the Structured APIs, but you might be trying to perform some custom logic on executors with UDFs or RDDs and either the task that you’re trying to serialize to these executors or the data you are trying to share cannot be serialized. This often happens when you’re working with either some code or data that cannot be serialized into a UDF or function, or if you’re working with strange data types that cannot be serialized. If you are using (or intend to be using Kryo serialization), verify that you’re actually registering your classes so that they are indeed serialized.

1. Try not to refer to any fields of the enclosing object in your UDFs when creating UDFs inside a Java or Scala class. This can cause Spark to try to serialize the whole enclosing object, which may not be possible. Instead, copy the relevant fields to local variables in the same scope as closure and use those.

# Indirect Performance Enhancements

## Design Choices
1. As we mentioned numerous times, Spark’s Structured APIs are consistent across languages in terms of speed and stability. That means that you should code with whatever language you are most comfortable using or is best suited for your use case.
1. These might manifest themselves as RDD transformations or user-defined functions (UDFs). If you’re going to do this, R and Python are not necessarily the best choice simply because of how this is actually executed. It’s also more difficult to provide stricter guarantees of types and manipulations when you’re defining functions that jump across languages. We find that using Python for the majority of the application, and porting some of it to Scala or writing specific UDFs in Scala as your application evolves, is a powerful technique—it allows for a nice balance between overall usability, maintainability, and performance.


## DATAFRAMES VERSUS SQL VERSUS DATASETS VERSUS RDDS
Across all languages, DataFrames, Datasets, and SQL are equivalent in speed. This means that if you’re using DataFrames in any of these languages, performance is equal. However, if you’re going to be defining UDFs, you’ll take a performance hit writing those in Python or R, and to some extent a lesser performance hit in Java and Scala. 
If you want to optimize for pure performance, it would behoove you to try and get back to DataFrames and SQL as quickly as possible. Although all DataFrame, SQL, and Dataset code compiles down to RDDs, Spark’s optimization engine will write “better” RDD code than you can manually and certainly do it with orders of magnitude less effort. Additionally, you will lose out on new optimizations that are added to Spark’s SQL engine every release.

1. Lastly, if you want to use RDDs, we definitely recommend using Scala or Java. If that’s not possible, we recommend that you restrict the “surface area” of RDDs in your application to the bare minimum. That’s because when Python runs RDD code, it’s serializes a lot of data to and from the Python process. This is very expensive to run over very big data and can also decrease stability.


## Object Serialization in RDDs
When you’re working with custom data types, you’re going to want to serialize them using Kryo because it’s both more compact and much more efficient than Java serialization. However, this does come at the inconvenience of registering the classes that you will be using in your application.

## DYNAMIC ALLOCATION
Spark provides a mechanism to dynamically adjust the resources your application occupies based on the workload. This means that your application can give resources back to the cluster if they are no longer used, and request them again later when there is demand. This feature is particularly useful if multiple applications share resources in your Spark cluster. This feature is disabled by default.

## Scheduling
Over the course of the previous chapters, we discussed a number of different potential optimizations that you can take advantage of to either help Spark jobs run in parallel with scheduler pools or help Spark applications run in parallel with something like dynamic allocation or setting max-executor-cores. 
Scheduling optimizations do involve some research and experimentation, and unfortunately there are not super-quick fixes beyond setting spark.scheduler.mode to FAIR to allow better sharing of resources across multiple users, or setting --max-executor-cores, which specifies the maximum number of executor cores that your application will need. Specifying this value can ensure that your application does not take up all the resources on the cluster. You can also change the default, depending on your cluster manager, by setting the configuration spark.cores.max to a default of your choice. Cluster managers also provide some scheduling primitives that can be helpful when optimizing multiple Spark Applications.

## Data at Rest
More often that not, when you’re saving data it will be read many times as other folks in your organization access the same datasets in order to run different analyses. Making sure that you’re storing your data for effective reads later on is absolutely essential to successful big data projects. This involves choosing your storage system, choosing your data format, and taking advantage of features such as data partitioning in some storage formats.

## FILE-BASED LONG-TERM DATA STORAGE
There are a number of different file formats available, from simple comma-separated values (CSV) files and binary blobs, to more sophisticated formats like Apache Parquet. One of the easiest ways to optimize your Spark jobs is to follow best practices when storing data and choose the most efficient storage format possible.

Generally you should always favor structured, binary types to store your data, especially when you’ll be accessing it frequently. Although files like “CSV” seem well-structured, they’re very slow to parse, and often also full of edge cases and pain points. For instance, improperly escaped new-line characters can often cause a lot of trouble when reading a large number of files. The most efficient file format you can generally choose is Apache Parquet. Parquet stores data in binary files with column-oriented storage, and also tracks some statistics about each file that make it possible to quickly skip data not needed for a query. It is well integrated with Spark through the built-in Parquet data source.

## SPLITTABLE FILE TYPES AND COMPRESSION
1. Whatever file format you choose, you should make sure it is “splittable”, which means that different tasks can read different parts of the file in parallel. When we read in the file, all cores were able to do part of the work. That’s because the file was splittable. If we didn’t use a splittable file type—say something like a malformed JSON file—we’re going to need to read in the entire file on a single machine, greatly reducing parallelism.

1. The main place splittability comes in is compression formats. A ZIP file or TAR archive cannot be split, which means that even if we have 10 files in a ZIP file and 10 cores, only one core can read in that data because we cannot parallelize access to the ZIP file. This is a poor use of resources. In contrast, files compressed using gzip, bzip2, or lz4 are generally splittable if they were written by a parallel processing framework like Hadoop or Spark. For your own input data, the simplest way to make it splittable is to upload it as separate files, ideally each no larger than a few hundred megabytes.

## TABLE PARTITIONING
Table partitioning refers to storing files in separate directories based on a key, such as the date field in the data. Storage managers like Apache Hive support this concept, as do many of Spark’s built-in data sources. Partitioning your data correctly allows Spark to skip many irrelevant files when it only requires data with a specific range of keys. For instance, if users frequently filter by “date” or “customerId” in their queries, partition your data by those columns. This will greatly reduce the amount of data that end users must read by most queries, and therefore dramatically increase speed.

The one downside of partitioning, however, is that if you partition at too fine a granularity, it can result in many small files, and a great deal of overhead trying to list all the files in the storage system.

## BUCKETING
The essense is that bucketing your data allows Spark to “pre-partition” data according to how joins or aggregations are likely to be performed by readers. This can improve performance and stability because data can be consistently distributed across partitions as opposed to skewed into just one or two. For instance, if joins are frequently performed on a column immediately after a read, you can use bucketing to ensure that the data is well partitioned according to those values. This can help prevent a shuffle before a join and therefore help speed up data access. Bucketing generally works hand-in-hand with partitioning as a second way of physically splitting up data.

## THE NUMBER OF FILES
In addition to organizing your data into buckets and partitions, you’ll also want to consider the number of files and the size of files that you’re storing. If there are lots of small files, you’re going to pay a price listing and fetching each of those individual files. For instance, if you’re reading a data from HDFS, this data is managed in blocks that are up to 128 MB in size (by default). This means if you have 30 files, of 5 MB each, you’re going to have to potentially request 30 blocks, even though the same data could have fit into 2 blocks (150 MB total).

Although there is not necessarily a panacea for how you want to store your data, the trade-off can be summarized as such. Having lots of small files is going to make the scheduler work much harder to locate the data and launch all of the read tasks. This can increase the network and scheduling overhead of the job. Having fewer large files eases the pain off the scheduler but it will also make tasks run longer. In this case, though, you can always launch more tasks than there are input files if you want more parallelism—Spark will split each file across multiple tasks assuming you are using a splittable format. 
In general, we recommend sizing your files so that they each contain at least a few tens of megatbytes of data.

One way of controlling data partitioning when you write your data is through a write option. To control how many records go into each file, you can specify the maxRecordsPerFile option to the write operation.

## DATA LOCALITY
Another aspect that can be important in shared cluster environments is data locality. Data locality basically specifies a preference for certain nodes that hold certain data, rather than having to exchange these blocks of data over the network. If you run your storage system on the same nodes as Spark, and the system supports locality hints, Spark will try to schedule tasks close to each input block of data. For example HDFS storage provides this option. There are several configurations that affect locality, but it will generally be used by default if Spark detects that it is using a local storage system. You will also see data-reading tasks marked as “local” in the Spark web UI.

## STATISTICS COLLECTION
Spark includes a cost-based query optimizer that plans queries based on the properties of the input data when using the structured APIs. However, to allow the cost-based optimizer to make these sorts of decisions, you need to collect (and maintain) statistics about your tables that it can use. There are two kinds of statistics: table-level and column-level statistics. Statistics collection is available only on named tables, not on arbitrary DataFrames or RDDs.

## Shuffle Configurations
Configuring Spark’s external shuffle service can often increase performance because it allows nodes to read shuffle data from remote machines even when the executors on those machines are busy (e.g., with garbage collection). This does come at the cost of complexity and maintenance, however, so it might not be worth it in your deployment. Beyond configuring this external service, there are also a number of configurations for shuffles, such as the number of concurrent connections per executor, although these usually have good defaults.

In addition, for RDD-based jobs, the serialization format has a large impact on shuffle performance—always prefer Kryo over Java serialization. Furthermore, for all jobs, the number of partitions of a shuffle matters. If you have too few partitions, then too few nodes will be doing work and there may be skew, but if you have too many partitions, there is an overhead to launching each one that may start to dominate. **Try to aim for at least a few tens of megabytes of data per output partition in your shuffle.**

## Memory Pressure and Garbage Collection
During the course of running Spark jobs, the executor or driver machines may struggle to complete their tasks because of a lack of sufficient memory or “memory pressure.” This may occur when an application takes up too much memory during execution or when garbage collection runs too frequently or is slow to run as large numbers of objects are created in the JVM and subsequently garbage collected as they are no longer used. One strategy for easing this issue is to ensure that you’re using the Structured APIs as much as possible. These will not only increase the efficiency with which your Spark jobs will execute, but it will also greatly reduce memory pressure because JVM objects are never realized and Spark SQL simply performs the computation on its internal format.

The Spark documentation includes some great pointers on tuning garbage collection for RDD and UDF based applications, and we paraphrase the following sections from that information.

### MEASURING THE IMPACT OF GARBAGE COLLECTION
The first step in garbage collection tuning is to gather statistics on how frequently garbage collection occurs and the amount of time it takes. You can do this by adding -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps to Spark’s JVM options using the spark.executor.extraJavaOptions configuration parameter. The next time you run your Spark job, you will see messages printed in the worker’s logs each time a garbage collection occurs. These logs will be on your cluster’s worker nodes (in the stdout files in their work directories), not in the driver.

### GARBAGE COLLECTION TUNING
To further tune garbage collection, you first need to understand some basic information about memory management in the JVM:

1. Java heap space is divided into two regions: Young and Old. The Young generation is meant to hold short-lived objects whereas the Old generation is intended for objects with longer lifetimes.

1. The Young generation is further divided into three regions: Eden, Survivor1, and Survivor2.

Here’s a simplified description of the garbage collection procedure:

1. When Eden is full, a minor garbage collection is run on Eden and objects that are alive from Eden and Survivor1 are copied to Survivor2.

1. The Survivor regions are swapped.

1. If an object is old enough or if Survivor2 is full, that object is moved to Old.

1. Finally, when Old is close to full, a full garbage collection is invoked. This involves tracing through all the objects on the heap, deleting the unreferenced ones, and moving the others to fill up unused space, so it is generally the slowest garbage collection operation.

The goal of garbage collection tuning in Spark is to ensure that only long-lived cached datasets are stored in the Old generation and that the Young generation is sufficiently sized to store all short-lived objects. This will help avoid full garbage collections to collect temporary objects created during task execution. Here are some steps that might be useful.

1. Gather garbage collection statistics to determine whether it is being run too often. If a full garbage collection is invoked multiple times before a task completes, it means that there isn’t enough memory available for executing tasks, so you should decrease the amount of memory Spark uses for caching (spark.memory.fraction).

1. If there are too many minor collections but not many major garbage collections, allocating more memory for Eden would help. You can set the size of the Eden to be an over-estimate of how much memory each task will need. If the size of Eden is determined to be E, you can set the size of the Young generation using the option -Xmn=4/3*E. (The scaling up by 4/3 is to account for space used by survivor regions, as well.)

As an example, if your task is reading data from HDFS, the amount of memory used by the task can be estimated by using the size of the data block read from HDFS. Note that the size of a decompressed block is often two or three times the size of the block. So if you want to have three or four tasks’ worth of working space, and the HDFS block size is 128 MB, we can estimate size of Eden to be 43,128 MB.

Try the G1GC garbage collector with -XX:+UseG1GC. It can improve performance in some situations in which garbage collection is a bottleneck and you don’t have a way to reduce it further by sizing the generations. Note that with large executor heap sizes, it can be important to increase the G1 region size with -XX:G1HeapRegionSize.

Monitor how the frequency and time taken by garbage collection changes with the new settings.

Our experience suggests that the effect of garbage collection tuning depends on your application and the amount of memory available. There are many more tuning options described online, but at a high level, managing how frequently full garbage collection takes place can help in reducing the overhead. You can specify garbage collection tuning flags for executors by setting spark.executor.extraJavaOptions in a job’s configuration.


# Direct Performance Enhancements

## Parallelism
The first thing you should do whenever trying to speed up a specific stage is to increase the degree of parallelism. In general, we recommend having at least two or three tasks per CPU core in your cluster if the stage processes a large amount of data. You can set this via the spark.default.parallelism property as well as tuning the spark.sql.shuffle.partitions according to the number of cores in your cluster.

## Improved Filtering
Another frequent source of performance enhancements is moving filters to the earliest part of your Spark job that you can. Sometimes, these filters can be pushed into the data sources themselves and this means that you can avoid reading and working with data that is irrelevant to your end result. Enabling partitioning and bucketing also helps achieve this. Always look to be filtering as much data as you can early on, and you’ll find that your Spark jobs will almost always run faster.

## Repartitioning and Coalescing
Repartition calls can incur a shuffle. However, doing some can optimize the overall execution of a job by balancing data across the cluster, so they can be worth it. In general, you should try to shuffle the least amount of data possible. For this reason, if you’re reducing the number of overall partitions in a DataFrame or RDD, first try coalesce method, which will not perform a shuffle but rather merge partitions on the same node into one partition. The slower repartition method will also shuffle data across the network to achieve even load balancing. Repartitions can be particularly helpful when performing joins or prior to a cache call. Remember that repartitioning is not free, but it can improve overall application performance and parallelism of your jobs.

## CUSTOM PARTITIONING
If your jobs are still slow or unstable, you might want to explore performing custom partitioning at the RDD level. This allows you to define a custom partition function that will organize the data across the cluster to a finer level of precision than is available at the DataFrame level. This is very rarely necessary, but it is an option. 

## User-Defined Functions (UDFs)
In general, avoiding UDFs is a good optimization opportunity. UDFs are expensive because they force representing data as objects in the JVM and sometimes do this multiple times per record in a query. You should try to use the Structured APIs as much as possible to perform your manipulations simply because they are going to perform the transformations in a much more efficient manner than you can do in a high-level language. There is also ongoing work to make data available to UDFs in batches, such as the Vectorized UDF extension for Python that gives your code multiple records at once using a Pandas data frame. 

## Temporary Data Storage (Caching)
In applications that reuse the same datasets over and over, one of the most useful optimizations is caching. Caching will place a DataFrame, table, or RDD into temporary storage (either memory or disk) across the executors in your cluster, and make subsequent reads faster. Although caching might sound like something we should do all the time, it’s not always a good thing to do. That’s because caching data incurs a serialization, deserialization, and storage cost. For example, if you are only going to process a dataset once (in a later transformation), caching it will only slow you down.

The use case for caching is simple: as you work with data in Spark, either within an interactive session or a standalone application, you will often want to reuse a certain dataset (e.g., a DataFrame or RDD). For example, in an interactive data science session, you might load and clean your data and then reuse it to try multiple statistical models. Or in a standalone application, you might run an iterative algorithm that reuses the same dataset. You can tell Spark to cache a dataset using the cache method on DataFrames or RDDs.

**Caching is a lazy operation, meaning that things will be cached only as they are accessed.** The RDD API and the Structured API differ in how they actually perform caching, so let’s review the gory details before going over the storage levels. When we cache an RDD, we cache the actual, physical data (i.e., the bits). The bits. When this data is accessed again, Spark returns the proper data. This is done through the RDD reference. However, in the Structured API, caching is done based on the physical plan. This means that we effectively store the physical plan as our key (as opposed to the object reference) and perform a lookup prior to the execution of a Structured job. This can cause confusion because sometimes you might be expecting to access raw data but because someone else already cached the data, you’re actually accessing their cached version. Keep that in mind when using this feature.

There are different storage levels that you can use to cache your data, specifying what type of storage to use.

Because caching itself is lazy—the data is cached only on the first time you run an action on the DataFrame.

## Joins
Joins are a common area for optimization. The biggest weapon you have when it comes to optimizing joins is simply educating yourself about what each join does and how it’s performed. This will help you the most. Additionally, equi-joins are the easiest for Spark to optimize at this point and therefore should be preferred wherever possible. Beyond that, simple things like trying to use the filtering ability of inner joins by changing join ordering can yield large speedups. Additionally, using broadcast join hints can help Spark make intelligent planning decisions when it comes to creating query plans. 
Avoiding Cartesian joins or even full outer joins is often low-hanging fruit for stability and optimizations because these can often be optimized into different filtering style joins when you look at the entire data flow instead of just that one particular job area. Lastly, following some of the other sections in this chapter can have a significant effect on joins. For example, collecting statistics on tables prior to a join will help Spark make intelligent join decisions. Additionally, bucketing your data appropriately can also help Spark avoid large shuffles when joins are performed.

## Aggregations
For the most part, there are not too many ways that you can optimize specific aggregations beyond filtering data before the aggregation having a sufficiently high number of partitions. However, if you’re using RDDs, controlling exactly how these aggregations are performed (e.g., using reduceByKey when possible over groupByKey) can be very helpful and improve the speed and stability of your code.

## Broadcast Variables
The basic premise is that if some large piece of data will be used across multiple UDF calls in your program, you can broadcast it to save just a single read-only copy on each node and avoid re-sending this data with each job. For example, broadcast variables may be useful to save a lookup table or a machine learning model. You can also broadcast arbitrary objects by creating broadcast variables using your SparkContext, and then simply refer to those variables in your tasks.


## Others
If you’re using the structured APIs, your code should run just about as fast as if you had written it in Scala, except if you’re not using UDFs in Python. If you’re using a UDF, you may have a performance impact. Refer back to Chapter 6 for more information on why this is the case.

If you’re using the unstructured APIs, specifically RDDs, then your performance is going to suffer (at the cost of a bit more flexibility). We touch on this reasoning in Chapter 12, but the fundamental idea is that Spark is going to have to work a lot harder converting information from something that Spark and the JVM can understand to Python and back again. This includes both functions as well as data and is a process known as serialization. We’re not saying it never makes sense to use them; it’s just something to be aware of when doing so.



















