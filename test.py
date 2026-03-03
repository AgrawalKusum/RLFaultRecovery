from controllers.nominal_walker import NominalWalker

def main():
    walker = NominalWalker(
        model_path="models/final_model.zip",
        stats_path="models/vec_normalize.pkl",
        render=True,
    )

    reward = walker.run_episode()
    print("Reward:", reward)

    walker.close()

if __name__ == "__main__":
    main()