import { signInWithEmailAndPassword } from "firebase/auth";
import{createUserWithEmailAndPassword} from "firebase/auth";
import { useHistory } from "react-router-dom";

import React, { useState } from "react";
import './loginsignup.css';
import { auth } from "../../firebase";

const LoginSignup = () => {
    const [action, setAction] = useState("Sign up");
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [signlog, setSignlog] = useState(false);

    const signIn = () => {
        if (signlog) {
            signInWithEmailAndPassword(auth, email, password)
                .then((userCredential) => {
                    console.log(userCredential);
                }).catch((error) => {
                    console.log(error);
                });
        } else {
            setSignlog(!signlog);
            setAction("Login");
        }
    }
    const signUp = () => {
        if (!signlog) {
            createUserWithEmailAndPassword(auth, email, password)
                .then((userCredential) => {
                    console.log(userCredential);
                    history.push('/');
                }).catch((error) => {
                    console.log(error);
                });
        } else {
            setSignlog(!signlog);
            setAction("Sign up");
        }
    }
    return (
        <div className="container"> 
            <div className="header">
                <div className="text">
                    {action}
                </div>
                <div className="underline"></div>
            </div>
            <div className="inputs">
                {action === "Login" ? <div></div> : <div className="input">
                    <input type="text" placeholder="   Username"/>
                </div>}
                <div className="input">
                    <input type="email" placeholder="   Email" value={email} onChange={(e) => setEmail(e.target.value)}/>
                </div>
                <div className="input">
                    <input type="password" placeholder="   Password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
                {action === "Sign up" ? <div></div> : <div className="forgot-password">Lost Password? <span>Click here!</span></div>}
                <div className="submit-container">
                    <div className={action === "Login" ? "submit gray" : "submit"} onClick={() => signUp()}>
                        <button type="button" className={action === "Login" ? "submit gray" : "submit"}>Sign up</button>
                    </div>
                    <div className={action === "Sign up" ? "submit gray" : "submit"} onClick={signIn}>
                        <button type="button" className={action === "Sign up" ? "submit gray" : "submit"}>Login</button> 
                    </div>
                </div>
            </div>
            
        </div>
    )
}

export default LoginSignup;
