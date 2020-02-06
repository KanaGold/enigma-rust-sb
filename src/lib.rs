// Built-In Attributes
#![no_std]

// Imports
extern crate eng_wasm;
extern crate eng_wasm_derive;
extern crate serde;
extern crate std;
extern crate statistical;
extern crate linregress;
extern crate rusty_machine;
//extern crate randomforests;this library does not work in Enigma environment

use eng_wasm::*;
use eng_wasm_derive::pub_interface;
use serde::{Serialize, Deserialize};
use std::f64;
use std::format;
use std::vec;
use statistical::{mean, standard_deviation,population_variance};
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder, RegressionParameters};

use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
//use randomforests::{RandomForest,TreeConfig,Value,Dataset,Item};

// Encrypted state keys
static DATAFORMS: &str = "dataforms";
// Structs
#[derive(Serialize, Deserialize)]
pub struct Dataform {
    index: U256,
    name: String,
    sex: U256,
    weight: U256,
    height: U256,
    age: U256,
}

// Public struct Contract which will consist of private and public-facing secret contract functions
pub struct Contract;

// Private functions accessible only by the secret contract
impl Contract {

    //get secret state
    fn get_datas() -> Vec<Dataform> {
        read_state!(DATAFORMS).unwrap_or_default()
    }

    //get average for weight
    fn get_mean() -> U256 {
        let dataform = Self::get_datas();
        let datanum = dataform.len();
        let mut v:Vec<f64> = Vec::with_capacity(datanum);
        for i in dataform {
            v.push(f64::from(i.weight.as_u32()));
        };
        U256::from(mean(&v).round() as u32)
    }

    //get stdev for weight
    fn get_stdev() -> U256 {
        let dataform = Self::get_datas();
        let datanum = dataform.len();
        let mut v:Vec<f64> = Vec::with_capacity(datanum);
        for i in dataform {
            v.push(f64::from(i.weight.as_u32()));
        };
        U256::from(standard_deviation(&v,None).round() as u32)
    }

    //get variance for weight
    fn get_variance() -> U256 {
        let dataform = Self::get_datas();
        let datanum = dataform.len();
        let mut v:Vec<f64> = Vec::with_capacity(datanum);
        for i in dataform {
            v.push(f64::from(i.weight.as_u32()));
        };
        U256::from(population_variance(&v,None).round() as u32)
    }

    //get max for weight, max for float type is a bit complicated
    fn get_max() -> U256 {
        match Self::get_datas().iter().max_by_key(|&m| m.weight) {
            Some(data) => {
                data.weight
            },
            None => U256::zero(),
        }
    }

    //get parameters for linear regression
    fn get_linregress() -> RegressionParameters { // return linregress::RegressionParameter
        let dataform = Self::get_datas();
        let datanum = dataform.len();
        let mut sex:Vec<f64> = Vec::with_capacity(datanum);
        let mut age:Vec<f64> = Vec::with_capacity(datanum);
        let mut height:Vec<f64> = Vec::with_capacity(datanum);
        let mut weight:Vec<f64> = Vec::with_capacity(datanum);
        for i in dataform {
            sex.push(f64::from(i.sex.as_u32()));
            age.push(f64::from(i.age.as_u32()));
            height.push(f64::from(i.height.as_u32()));
            weight.push(f64::from(i.weight.as_u32()));
        };
        let datafigure = std::vec![("Y",weight.clone()),("X1",age),("X2",height)];
        let datamodel = RegressionDataBuilder::new().build_from(datafigure).expect("Cannot build new regression.");
        let formula = "Y ~ X1 + X2";
        let model = FormulaRegressionBuilder::new()
                .data(&datamodel)
                .formula(formula)
                .fit().expect("Cannot build new model.");
        model.parameters
    }

    //find if the name exists in the state
    fn find_name(name:String) -> String {
        match Self::get_datas().iter().find(|&m| m.name == name) {
            Some(data) => {
                "Exists".to_string()
            },
            None => "None".to_string(),
        }
    }

    //sort the names and get the last one in dictionary order
    fn get_max_name() -> String {
        match Self::get_datas().iter().max_by_key(|&m| &m.name) {
            Some(data) => {
                data.name.to_string()
            },
            None => "None".to_string(),
        }
    }

    //get logistic regression
    fn get_logisticreg() -> String {
        let dataform = Self::get_datas();
        let datanum = dataform.len();
        let mut analyzedata:Vec<f64> = Vec::with_capacity(datanum*3);
        let mut categorydata:Vec<f64> = Vec::with_capacity(datanum);
        for i in dataform {
            categorydata.push(f64::from(i.sex.as_u32()));//sex feature is regarded as category
            analyzedata.push(f64::from(i.age.as_u32()));
            analyzedata.push(f64::from(i.height.as_u32()));
            analyzedata.push(f64::from(i.weight.as_u32()));
        };
        let inputs = Matrix::new(datanum,3,analyzedata);
        let targets = Vector::new(categorydata);
        let mut log_mod = LogisticRegressor::default();
        // Train the model
        log_mod.train(&inputs, &targets).unwrap();
        // Now we'll predict a new point
        let new_point = Matrix::new(1,3,vec![800.,1500.,630.]);//730 for 1, 530 for 0
        let output = log_mod.predict(&new_point).unwrap();
        (output[0]*100.0).to_string()
    }

    //get random forest result
    //randomforest does not work in Enigma environment. This might be due to the parallized computation functionality. other libraries for rust failed as well.
    /*
    fn get_randomforest() -> String {
        let mut dataset = Dataset::new();
        let mut item1 = Item::new();
        item1.insert("lang".to_string(), Value { data: "rust".to_string() });
        item1.insert("typing".to_string(), Value { data: "static".to_string() });
        dataset.push(item1);
        let mut item2 = Item::new();
        item2.insert("lang".to_string(), Value { data: "python".to_string() } );
        item2.insert("typing".to_string(), Value { data: "dynamic".to_string() } );
        dataset.push(item2);
        let mut item3 = Item::new();
        item3.insert("lang".to_string(), Value { data: "haskell".to_string() });
        item3.insert("typing".to_string(), Value { data: "static".to_string() });
        dataset.push(item3);
        let mut config = TreeConfig::new();
        config.decision = "lang".to_string();
        let forest = RandomForest::build("lang".to_string(), config, &dataset, 10, 3);//this line do not work
        let mut question = Item::new();
        question.insert("typing".to_string(), Value { data: "static".to_string() });
        let answer = RandomForest::predict(forest, question);
        let tester = Value{data: "haskell".to_string()};
        answer.get(&tester).expect("something wrong").to_string()
    }
    */
}

// Public trait defining public-facing secret contract functions
#[pub_interface]
pub trait ContractInterface{
    fn add_jsondata(data: String);
    fn return_latest_jsondata() -> String;
}

// Implementation of the public-facing secret contract functions defined in the ContractInterface
// trait implementation for the Contract struct above
impl ContractInterface for Contract {


    #[no_mangle]
    //initialize the dataset with the following sample data.
    fn add_jsondata(data: String) {
        let part:u32 = data.to_string().parse().unwrap_or_default();
        let mut index:U256 = U256::from(0);
        let mut datas = Self::get_datas();
        let datanumparam:i32 = data.parse().unwrap_or_default();
        for i in 1..datanumparam {
            let height:U256 = U256::from(1000+i*100);
            for j in 1..6 {
                let age:U256 = U256::from(500+j*50);
                for k in 0..2 {
                    let sex:U256 = U256::from(k);
                    for m in 0..2 {
                        let rand10:U256 = U256::from(m*100);
                        let weight:U256 = U256::from((f64::from(height.as_u32()) * 0.3 + f64::from(age.as_u32()) * 0.1 + f64::from(sex.as_u32()) * 200.0 + f64::from(rand10.as_u32())).round() as u32);
                        
                        index = U256::from(index.as_u32() + 1);
                        let name:String = "NAME".to_string() + &index.to_string();
                        let dataform = Dataform{index:index,name:name,sex:sex,weight:weight,height:height,age:age};
                        datas.push(dataform);
                    };
                };
            };
        };
        
        write_state!(DATAFORMS => datas);
    }
    #[no_mangle]
    //get the statistical computation result.
    //all but one line should be comment out.
    fn return_latest_jsondata() -> String {
        //Self::get_mean().to_string()
        //Self::get_stdev().to_string()
        //Self::get_variance().to_string()
        //Self::get_max().to_string()
        //Self::find_name("NAME111".to_string())
        //Self::get_max_name()
        Self::get_logisticreg()
        //Self::get_randomforest()//this line does not work
/*
        {
            let regression_parameters = Self::get_linregress();
            format!("Intercept Value: {}, Age: {}, Height: {}", regression_parameters.intercept_value, regression_parameters.regressor_values[0], regression_parameters.regressor_values[1]).to_string()
        }
*/
    }
}
